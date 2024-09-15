# Copyright 2023 PKU-Alignment Team. All Rights Reserved.
# Copyright 2023 Javier Rando (ETH Zurich). All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Trainer base class."""

from __future__ import annotations

import abc
import argparse
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, ClassVar

import deepspeed
import torch
import torch.distributed as dist
from transformers import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from safe_rlhf.logger import Logger
from safe_rlhf.utils import is_main_process


class TrainerBase(metaclass=abc.ABCMeta):
    """Trainer base class.

    Abstract methods:
        init_models: Initialize model and tokenizer.
        init_datasets: Initialize training and evaluation datasets.
        init_engines: Initialize DeepSpeed engines.
        train: Train model.
        set_train: Set training mode for all models.
    """

    TRAINING_TYPE: ClassVar[str]

    tokenizer: PreTrainedTokenizerBase

    args: argparse.Namespace
    logger: Logger

    @abc.abstractmethod
    def init_models(self) -> None:
        """Initialize model and tokenizer."""
        raise NotImplementedError

    @abc.abstractmethod
    def init_datasets(self) -> None:
        """Initialize training and evaluation datasets."""
        raise NotImplementedError

    @abc.abstractmethod
    def init_engines(self) -> None:
        """Initialize DeepSpeed engines."""
        raise NotImplementedError

    def init_logger(self) -> None:
        """Set logger."""
        if self.args.log_type is None:
            self.logger = Logger()

        time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        self.args.log_dir = self.args.log_dir or self.args.output_dir
        self.args.log_project = self.args.log_project or "safe-rlhf"
        self.args.log_run_name = (
            self.args.log_run_name or f"{self.TRAINING_TYPE}-{time}"
        )

        self.logger = Logger(
            log_type=self.args.log_type,
            log_dir=self.args.log_dir,
            log_project=self.args.log_project,
            log_run_name=self.args.log_run_name,
        )

    @abc.abstractmethod
    def train(self) -> None:
        """Train model."""
        raise NotImplementedError

    def eval(self) -> dict[str, torch.Tensor]:
        """Evaluate model."""
        return {}

    @abc.abstractmethod
    def set_train(self, mode: bool = True) -> None:
        """Set training mode for all models."""
        raise NotImplementedError

    def set_eval(self) -> None:
        """Set model to evaluation mode."""
        self.set_train(mode=False)

def save(
    self,
    model: deepspeed.DeepSpeedEngine | None = None,
    ds_config: dict[str, Any] | None = None,
) -> None:
    """Save model in PyTorch format without saving DeepSpeed checkpoint."""
    dist.barrier()

    if model is None:
        model = self.model  # pylint: disable=no-member
    if ds_config is None:
        ds_config = self.ds_config  # pylint: disable=no-member

    self.logger.print(f'Saving model to "{self.args.output_dir}" ...')

    # Get the underlying PyTorch model
    model_to_save: PreTrainedModel = getattr(model, "module", model)

    # Gather full model weights
    model_to_save = model.module if hasattr(model, 'module') else model
    state_dict = model_to_save.state_dict()
    if ds_config["zero_optimization"]["stage"] > 0:
        state_dict = model._zero3_consolidated_16bit_state_dict()

    # Save model in PyTorch format
    if is_main_process():
        self.logger.print("Saving PyTorch model...")
        torch.save(state_dict, os.path.join(self.args.output_dir, WEIGHTS_NAME))
        model_to_save.config.save_pretrained(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)

    dist.barrier()
    self.logger.print("Model saved!")
