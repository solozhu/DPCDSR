import codecs
import random
from pathlib import Path

import torch


class DataLoader(object):
    """HGNCDSR-compatible data reader that keeps only one domain sequence."""

    def __init__(self, domains, batch_size, opt, evaluation, predict_domain="x", root=None):
        self.batch_size = batch_size
        self.opt = opt
        self.eval = evaluation
        self.domains = domains
        self.predict_domain = self._normalize_domain(predict_domain)
        self.root = Path(root or Path(__file__).resolve().parent)
        self.data_root = self.root / "dataset" / domains

        opt.source_item_num = self.read_item(self.data_root / "Alist.txt")
        opt.target_item_num = self.read_item(self.data_root / "Blist.txt")
        opt.user_num = self.read_user(self.data_root / "userlist.txt")
        opt.pad_id = opt.source_item_num + opt.target_item_num

        if "Enter" in domains:
            opt.maxlen = 30
            opt.L = 30
        else:
            opt.maxlen = 15
            opt.L = 15

        if evaluation < 0:
            raw_data, raw_users = self.read_sequence_data(self.data_root / "traindata_new.txt", train=True)
            data = self.preprocess_train(raw_data, raw_users)
        elif evaluation == 2:
            raw_data, raw_users = self.read_sequence_data(self.data_root / "validdata_new2.txt", train=False)
            data = self.preprocess_predict(raw_data, raw_users)
        else:
            raw_data, raw_users = self.read_sequence_data(self.data_root / "testdata_new2.txt", train=False)
            data = self.preprocess_predict(raw_data, raw_users)

        if evaluation == -1:
            random.shuffle(data)
            if batch_size > len(data):
                batch_size = len(data)
                self.batch_size = batch_size
            if data and len(data) % batch_size != 0:
                data += data[:batch_size]
            data = data[: (len(data) // batch_size) * batch_size] if data else []

        self.data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    @staticmethod
    def _normalize_domain(domain):
        key = str(domain).lower()
        if key in {"x", "a", "source"}:
            return "x"
        if key in {"y", "b", "target"}:
            return "y"
        raise ValueError("unknown domain %r, expected x/y" % domain)

    def in_domain(self, item):
        if self.predict_domain == "x":
            return 0 <= item < self.opt.source_item_num
        return self.opt.source_item_num <= item < self.opt.source_item_num + self.opt.target_item_num

    def sample_negative(self, positive):
        while True:
            if self.predict_domain == "x":
                sample = random.randint(0, self.opt.source_item_num - 1)
            else:
                sample = random.randint(self.opt.source_item_num, self.opt.source_item_num + self.opt.target_item_num - 1)
            if sample != positive:
                return sample

    def read_item(self, fname):
        with codecs.open(str(fname), "r", encoding="utf-8") as fr:
            return sum(1 for _ in fr)

    def read_user(self, fname):
        with codecs.open(str(fname), "r", encoding="utf-8") as fr:
            return sum(1 for _ in fr)

    def read_sequence_data(self, fname, train):
        data = []
        users = []
        with codecs.open(str(fname), "r", encoding="utf-8") as infile:
            for line in infile:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                users.append(int(parts[0]))
                pairs = []
                for value in parts[2:]:
                    fields = value.split("|")
                    if len(fields) < 2:
                        continue
                    item, timestamp = fields[0], fields[1]
                    pairs.append((int(item), int(timestamp)))
                pairs.sort(key=lambda x: x[1])
                items = [item for item, _ in pairs]
                data.append(items if train else (items[:-1], items[-1]))
        return data, users

    def make_position(self, seq):
        pos = [0] * len(seq)
        idx = 0
        for i in range(len(seq) - 1, -1, -1):
            if seq[i] != self.opt.pad_id:
                idx += 1
                pos[i] = idx
        return pos

    def left_pad(self, seq):
        seq = seq[-self.opt.maxlen:]
        return [self.opt.pad_id] * (self.opt.maxlen - len(seq)) + seq

    def preprocess_train(self, raw_data, raw_users):
        processed = []
        for items, user in zip(raw_data, raw_users):
            domain_items = [item for item in items if self.in_domain(item)]
            if len(domain_items) < 2:
                continue
            ground = domain_items[-1]
            seq = self.left_pad(domain_items[:-1])
            position = self.make_position(seq)
            neg = [self.sample_negative(ground) for _ in range(self.opt.neg_samples)]
            processed.append([seq, [ground], position, user, neg])
        return processed

    def preprocess_predict(self, raw_data, raw_users):
        processed = []
        for (prefix, ground), user in zip(raw_data, raw_users):
            if not self.in_domain(ground):
                continue
            domain_prefix = [item for item in prefix if self.in_domain(item)]
            if not domain_prefix:
                continue
            seq = self.left_pad(domain_prefix)
            position = self.make_position(seq)
            neg = [self.sample_negative(ground) for _ in range(999)]
            processed.append([seq, [ground], position, user, neg])
        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise TypeError
        if key < 0:
            raise IndexError
        batch = list(zip(*self.data[key]))
        return (
            torch.LongTensor(batch[0]),
            torch.LongTensor(batch[1]),
            torch.LongTensor(batch[2]),
            torch.LongTensor(batch[3]),
            torch.LongTensor(batch[4]),
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
