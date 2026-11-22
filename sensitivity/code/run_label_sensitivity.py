#!/usr/bin/env python3
"""Rescore frozen predictions using ONLY the 101 accepted audit decisions.

The repository's metric functions and exact-string evaluation rule are reused.
Original data, training labels, prediction files, and repository code are not
modified. Outputs are a separate, explicitly scoped sensitivity analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repository_metadata(repo: Path) -> dict:
    """Read provenance from the supplied directory; do not infer a remote snapshot."""
    def git_value(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ['git', '-C', str(repo), *arguments],
                capture_output=True, text=True, check=False,
            )
        except FileNotFoundError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    # A downloaded ZIP may be stored inside an unrelated Git working tree.
    # Only attribute Git metadata when its top-level directory is this repo.
    top_level = git_value('rev-parse', '--show-toplevel')
    owns_git_metadata = bool(top_level and Path(top_level).resolve() == repo.resolve())
    commit = git_value('rev-parse', 'HEAD') if owns_git_metadata else None
    remote = git_value('remote', 'get-url', 'origin') if owns_git_metadata else None
    # Do not persist credentials embedded in an HTTP remote URL.
    if remote and remote.startswith(('https://', 'http://')):
        parts = urlsplit(remote)
        remote = urlunsplit((parts.scheme, parts.netloc.rsplit('@', 1)[-1],
                            parts.path, '', ''))
    status = git_value('status', '--porcelain', '--untracked-files=no') if owns_git_metadata else None
    return {
        'source_kind': 'git_checkout' if commit else 'directory_without_git_metadata',
        'git_commit': commit,
        'git_remote_origin': remote,
        'tracked_worktree_clean': status == '' if status is not None else None,
        'note': ('Metadata describes the supplied local directory. Input hashes below '
                 'identify the exact bytes used; this command does not fetch a remote.'),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--crosswalk', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--n-boot', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = args.output_dir.resolve()
    if out == repo or repo in out.parents:
        raise ValueError('Write sensitivity outputs outside the source repository.')
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo / 'src'))
    sys.path.insert(0, str(repo / 'scripts'))
    from trframenet_revision.metrics import (
        bootstrap_mean_ci, paired_bootstrap_diff, exact_mcnemar, safe_kappa,
    )
    from evaluate_primary import DISPLAY

    pred_path = repo / 'results/primary/primary_predictions.csv'
    train_path = repo / 'data/lexicon/train.csv'
    test_path = repo / 'data/lexicon/test.csv'
    audit_path = repo / 'data/resource_quality/audit_records.csv'
    pred = pd.read_csv(pred_path)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    audit = pd.read_csv(audit_path).fillna('')
    cw = pd.read_csv(args.crosswalk).fillna('')
    for name, frame, idcol in [('predictions', pred, 'ID'), ('train', train, 'ID'),
                               ('test', test, 'ID'), ('audit', audit, 'sample_id'),
                               ('crosswalk', cw, 'sample_id')]:
        if frame[idcol].isna().any() or not frame[idcol].is_unique:
            raise ValueError(f'{name} has missing or nonunique IDs.')
    assert len(pred) == len(test) == 1673
    assert len(train) == 6876
    assert set(train.ID).isdisjoint(test.ID)
    assert set(pred.ID) == set(test.ID)
    assert (pred.set_index('ID').FRAME1 == test.set_index('ID').loc[pred.ID].FRAME1).all()
    assert len(cw) == len(audit) == 400
    assert set(cw.sample_id) == set(audit.sample_id)
    assert cw.frozen_id.is_unique
    ai = audit.set_index('sample_id').loc[cw.sample_id]
    for a, b in [('final_outcome', 'adjudication_outcome'),
                 ('adjudicated_frame', 'adjudicated_frame'),
                 ('released_frame', 'released_frame')]:
        assert (cw[a].to_numpy() == ai[b].to_numpy()).all(), (a, b)
    frozen = pd.concat([train.assign(split='train'), test.assign(split='test')]).set_index('ID')
    assert set(cw.frozen_id).issubset(frozen.index)
    assert (cw.frozen_frame.to_numpy() == frozen.loc[cw.frozen_id, 'FRAME1'].to_numpy()).all()
    assert (cw.split.to_numpy() == frozen.loc[cw.frozen_id, 'split'].to_numpy()).all()
    targets = cw[cw.final_outcome == 'New label accepted'].copy()
    assert len(targets) == 101
    test_audit = cw[cw.split == 'test'].copy()
    target_test = targets[targets.split == 'test'].copy()
    assert len(test_audit) == 70 and len(target_test) == 15
    assert (target_test.adjudicated_frame != '').all()

    old = pred.FRAME1.copy()
    revised = old.copy()
    replacement = target_test.set_index('frozen_id').adjudicated_frame
    target_mask = pred.ID.isin(replacement.index)
    revised.loc[target_mask] = pred.loc[target_mask, 'ID'].map(replacement)
    changed = old != revised
    assert changed.sum() == 14
    assert not changed[~target_mask].any()
    # The 299 non-target records retain their exact frozen benchmark labels.
    cw['sensitivity_scope'] = np.where(cw.split == 'train', 'training_not_changed',
                             np.where(cw.final_outcome == 'New label accepted',
                                      'accepted_test_decision', 'test_label_preserved'))
    cw.to_csv(out / 'audit_400_scope.csv', index=False, encoding='utf-8-sig')
    target_test['literal_change'] = target_test.frozen_frame != target_test.adjudicated_frame
    target_test.to_csv(out / 'accepted_test_15_mapping.csv', index=False, encoding='utf-8-sig')

    gold_manifest = pred[['ID', 'SYNSET', 'DEFINITION']].copy()
    gold_manifest['gold_original'] = old
    gold_manifest['gold_sensitivity_101_only'] = revised
    gold_manifest['audited_test'] = pred.ID.isin(test_audit.frozen_id)
    gold_manifest['accepted_audit_decision'] = target_mask
    gold_manifest['literal_label_change'] = changed
    gold_manifest.to_csv(out / 'test_gold_comparison.csv', index=False, encoding='utf-8-sig')
    # Deliberately exclude stale original-gold candidate-membership flags.
    gold_manifest[pred.ID.isin(test_audit.frozen_id)].to_csv(
        out / 'audited_test_70_gold_comparison.csv', index=False, encoding='utf-8-sig')

    scopes = {'full_test': np.ones(len(pred), dtype=bool),
              'audited_test': pred.ID.isin(test_audit.frozen_id).to_numpy(),
              'accepted_audit_test': target_mask.to_numpy()}
    rows = []
    item_rows = []
    for scope, mask in scopes.items():
        for col, (category, model) in DISPLAY.items():
            valid = mask & pred[col].notna().to_numpy() & old.notna().to_numpy()
            po = pred.loc[valid, col].astype(str)
            go = old.loc[valid].astype(str)
            gn = revised.loc[valid].astype(str)
            before = (po == go).to_numpy()
            after = (po == gn).to_numpy()
            b, blo, bhi = bootstrap_mean_ci(before, n_boot=args.n_boot, seed=args.seed)
            a, alo, ahi = bootstrap_mean_ci(after, n_boot=args.n_boot, seed=args.seed)
            delta = paired_bootstrap_diff(after, before, n_boot=args.n_boot, seed=args.seed)
            mc = exact_mcnemar(after, before)
            rows.append({'scope': scope, 'prediction_column': col, 'category': category,
                         'model': model, 'n': int(valid.sum()),
                         'correct_original': int(before.sum()), 'correct_revised': int(after.sum()),
                         'accuracy_original': b, 'accuracy_revised': a,
                         'accuracy_original_ci_low': blo, 'accuracy_original_ci_high': bhi,
                         'accuracy_revised_ci_low': alo, 'accuracy_revised_ci_high': ahi,
                         'delta_pp': 100 * delta['difference'],
                         'delta_ci_low_pp': 100 * delta['ci_low'],
                         'delta_ci_high_pp': 100 * delta['ci_high'],
                         'gained_correct': mc['a_only'], 'lost_correct': mc['b_only'],
                         'annotation_change_mcnemar_p': mc['p_value'],
                         'kappa_original': safe_kappa(go, po),
                         'kappa_revised': safe_kappa(gn, po)})
            if scope == 'accepted_audit_test':
                for idx, bc, ac in zip(pred.index[valid], before, after):
                    item_rows.append({'ID': pred.at[idx, 'ID'], 'model': model,
                                      'gold_original': old.at[idx], 'gold_sensitivity': revised.at[idx],
                                      'prediction': pred.at[idx, col],
                                      'correct_original': bool(bc), 'correct_revised': bool(ac)})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / 'sensitivity_metrics.csv', index=False)
    pd.DataFrame(item_rows).to_csv(out / 'accepted_test_item_effects.csv', index=False)

    focus = ['lora_unconstrained_pred', 'lora_candidate_conditioned_pred',
             'encoder_mmpnet_linearsvc_pred', 'setup_a_gemini_2.5_flash_pred',
             'setup_a_gpt_4o_mini_pred', 'setup_a_gemma3_12b_pred']
    pairs = []
    for scope in ['full_test', 'audited_test']:
        for policy, gold in [('original', old), ('101_only', revised)]:
            for ca, cb in itertools.combinations(focus, 2):
                valid = scopes[scope] & pred[ca].notna().to_numpy() & pred[cb].notna().to_numpy()
                va = (pred.loc[valid, ca].astype(str) == gold.loc[valid].astype(str)).to_numpy()
                vb = (pred.loc[valid, cb].astype(str) == gold.loc[valid].astype(str)).to_numpy()
                pairs.append({'scope': scope, 'gold_policy': policy,
                              'model_a': DISPLAY[ca][1], 'model_b': DISPLAY[cb][1],
                              'n': int(valid.sum()),
                              **paired_bootstrap_diff(va, vb, n_boot=args.n_boot, seed=args.seed),
                              **exact_mcnemar(va, vb)})
    pd.DataFrame(pairs).to_csv(out / 'model_pairwise_tests.csv', index=False)

    full = metrics[metrics.scope == 'full_test']
    archived = pd.read_csv(repo / 'results/primary/primary_metrics_with_ci.csv')
    compare = full.merge(archived[['prediction_column', 'n', 'accuracy', 'kappa']],
                         on='prediction_column', validate='one_to_one', suffixes=('', '_archived'))
    assert (compare.n == compare.n_archived).all()
    assert np.allclose(compare.accuracy_original, compare.accuracy, atol=1e-12, rtol=0)
    assert np.allclose(compare.kappa_original, compare.kappa, atol=1e-12, rtol=0)
    compare[['prediction_column', 'n', 'accuracy_original', 'accuracy',
             'kappa_original', 'kappa']].to_csv(out / 'original_metrics_reproduction.csv', index=False)
    full_order_before = full.sort_values('accuracy_original', ascending=False).model.tolist()
    full_order_after = full.sort_values('accuracy_revised', ascending=False).model.tolist()

    inputs = [pred_path, train_path, test_path, audit_path, args.crosswalk.resolve(),
              repo / 'scripts/evaluate_primary.py', repo / 'src/trframenet_revision/metrics.py']
    manifest = {
        'source_policy': 'Frozen predictions and evaluation functions from the directory supplied by --repo; exact input bytes are identified by SHA-256.',
        'run_created_at_utc': datetime.now(timezone.utc).isoformat(),
        'repository': repository_metadata(repo),
        'runner_sha256': sha256(Path(__file__).resolve()),
        'comparison_rule': 'Exact string equality, unchanged from repository evaluator.',
        'gold_policy': 'Only New label accepted decisions for held-out test IDs replace gold in a separate sensitivity series; all predictions and training labels remain frozen.',
        'n_boot': args.n_boot, 'seed': args.seed,
        'counts': {'audit_total': 400, 'audit_train': 330, 'audit_test': 70,
                   'accepted_total': 101, 'accepted_train': 86, 'accepted_test': 15,
                   'literal_test_label_changes': 14, 'accepted_test_noops': 1,
                   'unchanged_audit_records_outside_101': 299},
        'original_metrics_reproduced_for_models': len(compare),
        'full_test_ranking_unchanged': full_order_before == full_order_after,
        'full_test_order_original': full_order_before, 'full_test_order_revised': full_order_after,
        'omitted_metrics': 'Conditional retrieval accuracy and Recall@k are not recomputed from stale original-gold flags.',
        'interpretation': 'The 70-item audited subset is diagnostic and not representative of the full test distribution. Full-test sensitivity changes only the audited accepted test labels, not all latent annotation errors. No retraining effect is measured.',
        'inputs': [{'file': str(x.relative_to(repo)) if repo in x.parents else x.name,
                    'sha256': sha256(x), 'bytes': x.stat().st_size} for x in inputs],
    }
    (out / 'run_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest['counts']))
    print(full[['model', 'n', 'accuracy_original', 'accuracy_revised', 'delta_pp']].to_string(index=False))
    print('\nAUDITED TEST')
    print(metrics[metrics.scope == 'audited_test'][['model', 'n', 'correct_original', 'correct_revised',
          'accuracy_original', 'accuracy_revised', 'delta_pp', 'delta_ci_low_pp', 'delta_ci_high_pp']].to_string(index=False))


if __name__ == '__main__':
    main()
