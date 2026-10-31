"""Loop region."""
from dataclasses import dataclass


@dataclass
class LoopRegion:
    start: float = 0.0
    end: float = 0.0
    enabled: bool = False

    def is_active(self):
        return self.enabled and self.end > self.start + 0.05

    def wrap(self, t):
        if not self.is_active():
            return t
        if t < self.end:
            return t
        if t < self.start:
            return t
        length = self.end - self.start
        return self.start + ((t - self.start) % length)

    def to_dict(self):
        return {"start": self.start, "end": self.end,
                "enabled": self.enabled}

    @classmethod
    def from_dict(cls, d):
        if not d:
            return cls()
        return cls(start=float(d.get("start", 0.0)),
                   end=float(d.get("end", 0.0)),
                   enabled=bool(d.get("enabled", False)))
