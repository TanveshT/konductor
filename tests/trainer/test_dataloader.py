from unittest.mock import MagicMock

import torch

from konductor.data import Split
from konductor.trainer.trainer import ZipDataloader, _get_dataloader_from_dataset_configs


def _make_fake_config(tensor: torch.Tensor):
    """Create a mock DatasetConfig whose get_dataloader returns a list of tensors."""
    cfg = MagicMock()
    cfg.get_dataloader.return_value = tensor
    return cfg


def test_single_dataloader_returns_unwrapped():
    data = torch.arange(10)
    cfg = _make_fake_config(data)
    result = _get_dataloader_from_dataset_configs([cfg], Split.TRAIN)
    assert result is data
    cfg.get_dataloader.assert_called_once_with(Split.TRAIN)


def test_multiple_dataloaders_returns_zip():
    t1 = torch.arange(5)
    t2 = torch.arange(3)
    cfg1 = _make_fake_config(t1)
    cfg2 = _make_fake_config(t2)
    result = _get_dataloader_from_dataset_configs([cfg1, cfg2], Split.VAL)
    assert isinstance(result, ZipDataloader)
    cfg1.get_dataloader.assert_called_once_with(Split.VAL)
    cfg2.get_dataloader.assert_called_once_with(Split.VAL)


def test_zip_dataloader_len_is_min():
    dl1 = list(range(5))
    dl2 = list(range(3))
    zipped = ZipDataloader([dl1, dl2])
    assert len(zipped) == 3


def test_zip_dataloader_iter():
    dl1 = [torch.tensor([1.0]), torch.tensor([2.0])]
    dl2 = [torch.tensor([3.0]), torch.tensor([4.0])]
    zipped = ZipDataloader([dl1, dl2])
    batches = list(zipped)
    assert len(batches) == 2
    assert torch.equal(batches[0][0], torch.tensor([1.0]))
    assert torch.equal(batches[0][1], torch.tensor([3.0]))
    assert torch.equal(batches[1][0], torch.tensor([2.0]))
    assert torch.equal(batches[1][1], torch.tensor([4.0]))


def test_zip_dataloader_getitem():
    dl1 = [torch.tensor([10.0]), torch.tensor([20.0])]
    dl2 = [torch.tensor([30.0]), torch.tensor([40.0])]
    zipped = ZipDataloader([dl1, dl2])
    item = zipped[1]
    assert torch.equal(item[0], torch.tensor([20.0]))
    assert torch.equal(item[1], torch.tensor([40.0]))
