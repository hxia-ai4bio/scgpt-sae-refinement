import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier
from sae_refinement.core import SAE, Ridge, relation
from sae_refinement.train import load_data


class Numerics(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_signed_raw_topk_and_fixed_mean(self):
        m = SAE(np.ones(512, np.float32), 40, 32, 42)
        with torch.no_grad():
            m.encoder.weight.zero_()
            m.encoder.bias.copy_(torch.arange(-40., 0.))
        h = torch.ones(3, 512)
        rec, z = m(h)
        self.assertTrue(torch.equal((z != 0).sum(1), torch.full((3,), 32)))
        self.assertTrue(torch.equal(z[0, 8:], torch.arange(-32., 0.)))
        self.assertTrue(torch.equal(z[:, :8], torch.zeros(3, 8)))
        self.assertTrue(torch.allclose(rec, m.decoder(z)+m.mean))
        self.assertNotIn('mean', dict(m.named_parameters()))
        torch.testing.assert_close(m.decoder.weight.norm(dim=0), torch.ones(40))

    def test_kl_identity_teacher_detached_and_both_student_sides(self):
        torch.manual_seed(1)
        h = torch.randn(8, 512, dtype=torch.float64, requires_grad=True)
        mean = torch.zeros(512, dtype=torch.float64)
        g = torch.arange(2).repeat_interleave(4)
        mask = g[:, None] != g[None, :]
        identity, _ = relation(h.detach(), h.detach(), mean, mask)
        self.assertLess(abs(identity.item()), 1e-12)
        q = (h.detach()+.3*torch.randn_like(h)).requires_grad_()
        anchors = (h.detach()+.2*torch.randn_like(h)).requires_grad_()
        kl, _ = relation(h, q, mean, mask, anchor_h=h, anchor_rec=anchors)
        kl.backward()
        self.assertIsNone(h.grad)
        self.assertGreater(q.grad.norm().item(), 0)
        self.assertGreater(anchors.grad.norm().item(), 0)

    def test_balanced_ridge_matches_sklearn(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(80, 12)); y = np.array(list(range(8))*5+[0]*40)
        t = rng.normal(size=(17, 12))
        ours = Ridge(x, y).fit(10.)
        ref = RidgeClassifier(alpha=10., class_weight='balanced', solver='cholesky')
        ref.fit((x-ours.mu)/ours.sd, y)
        np.testing.assert_allclose(ours.scores(t), ref.decision_function((t-ours.mu)/ours.sd), atol=1e-10)

    def test_reject_overlapping_and_leaking_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cache = root/'data.npz'; sp = root/'splits.json'
            rng = np.random.default_rng(3)
            np.savez(cache, embedding=rng.normal(size=(24, 512)), labels=np.tile(np.arange(8), 3), groups=np.repeat(['a','b','c'],8))
            split = dict(train=list(range(8)), valid=list(range(8,16)), test=list(range(16,24)))
            sp.write_text(json.dumps(split)); load_data(cache, sp, True)
            split['valid'][0] = 0; sp.write_text(json.dumps(split))
            with self.assertRaises(ValueError): load_data(cache, sp, True)
            split = dict(train=list(range(8)),valid=list(range(8,16)),test=list(range(16,24)))
            sp.write_text(json.dumps(split))
            np.savez(cache, embedding=rng.normal(size=(24,512)), labels=np.tile(np.arange(8),3), groups=np.repeat(['a','b','a'],8))
            with self.assertRaises(ValueError): load_data(cache, sp, True)


if __name__ == '__main__':
    unittest.main()
