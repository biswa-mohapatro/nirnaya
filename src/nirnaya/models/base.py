"""Architecture-neutral decision model interface."""

from abc import ABC, abstractmethod

from nirnaya.contracts import DecisionLogits


class BaseDecisionModel(ABC):
    """Common boundary consumed by training, inference, and RLCD."""

    @abstractmethod
    def forward(self, batch: object) -> DecisionLogits:
        """Return raw masked outcome logits for a backend-specific batch."""

    def __call__(self, batch: object) -> DecisionLogits:
        return self.forward(batch)
