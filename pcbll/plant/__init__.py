"""Plants: the true systems the surrogates are trained on and the controllers are tested against.

``PLANTS`` maps a name to a factory.  ``get_plant`` also resolves the
synthetic plant from its name pattern and the process-noise suffix.
"""

from pcbll.plant.base import Constraint, Controller, Plant, PlantSpec

__all__ = ["Constraint", "Controller", "Plant", "PlantSpec", "PLANTS", "get_plant"]


def _registry() -> dict:
    from pcbll.plant.cstr import PLANTS as cstr_plants

    return dict(cstr_plants)


class _Lazy(dict):
    """Registry that imports do-mpc only when first used."""

    _loaded = False

    def _load(self):
        if not self._loaded:
            self._loaded = True
            self.update(_registry())

    def __getitem__(self, key):
        self._load()
        return dict.__getitem__(self, key)

    def __contains__(self, key):
        self._load()
        return dict.__contains__(self, key)

    def __iter__(self):
        self._load()
        return dict.__iter__(self)

    def __len__(self):
        self._load()
        return dict.__len__(self)

    def keys(self):
        self._load()
        return dict.keys(self)


PLANTS = _Lazy()


def get_plant(name: str):
    """A plant by registry name, ``synthetic_a<setpoint>[_b<beta>][_u<umax>]`` for any values,
    and ``<base>_pn<level>`` for the base plant with additive process noise."""
    if "_pn" in name:
        from pcbll.plant.noisy import ProcessNoisePlant

        base, level = name.rsplit("_pn", 1)
        return ProcessNoisePlant(get_plant(base), float(level))
    if name in PLANTS:
        return PLANTS[name]()
    if name.startswith("synthetic_a"):
        from pcbll.plant.synthetic import SyntheticPlant

        rest = name[len("synthetic_a"):]
        kw = {}
        if "_u" in rest:
            rest, u = rest.split("_u")
            kw["u_max"] = float(u)
        if "_b" in rest:
            rest, b = rest.split("_b")
            kw["beta"] = float(b)
        return SyntheticPlant(setpoint=float(rest), **kw)
    raise KeyError(name)
