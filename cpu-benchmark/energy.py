"""Read package energy once per batch; never apportion it to worker processes."""
from pathlib import Path
import threading
import time


def counter_delta(previous, current, maximum):
    return (current - previous) % maximum


class PackageEnergy:
    def __init__(self, root=Path('/sys/class/powercap')):
        self.domains = {}
        # Resolve symlinks and deduplicate: class entries can include both the
        # top-level control type and individual package/subdomain links.
        paths = set()
        for control in root.glob('*'):
            if control.is_dir():
                paths.update(p.resolve() for p in control.rglob('energy_uj'))
        self.errors = []
        for path in sorted(paths):
            try:
                name = (path.parent / 'name').read_text().strip()
                if name.startswith('package-'):
                    maximum = int((path.parent / 'max_energy_range_uj').read_text())
                    int(path.read_text())
                    self.domains[str(path)] = (name, maximum)
            except (OSError, ValueError) as error:
                self.errors.append(str(error))
        self.stop_event = threading.Event()

    def metadata(self):
        return {'available': bool(self.domains), 'domains': self.domains,
                'errors': self.errors,
                'scope': 'all exposed CPU packages, including other processes; excludes DIMMs',
                'reason': None if self.domains else 'No readable package energy counter exposed'}

    def start(self):
        self.previous = {}
        self.total_uj = {p: 0 for p in self.domains}
        self.started = time.perf_counter()
        self.stop_event.clear()
        self.thread = None
        try:
            self.previous = {p: int(Path(p).read_text()) for p in self.domains}
        except (OSError, ValueError) as error:
            self.errors.append(str(error))
        if self.domains and not self.errors:
            self.thread = threading.Thread(target=self._poll, daemon=True)
            self.thread.start()

    def _sample(self):
        if self.errors:
            return
        for path, (_, maximum) in self.domains.items():
            current = int(Path(path).read_text())
            self.total_uj[path] += counter_delta(self.previous[path], current, maximum)
            self.previous[path] = current

    def _poll(self):
        try:
            while not self.stop_event.wait(0.25):
                self._sample()
        except (OSError, ValueError) as error:
            self.errors.append(str(error))

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        try:
            self._sample()
        except (OSError, ValueError) as error:
            self.errors.append(str(error))
        duration = time.perf_counter() - self.started
        joules = sum(self.total_uj.values()) / 1e6 if self.domains and not self.errors else None
        return {**self.metadata(), 'joules': joules,
                'average_watts': joules / duration if joules is not None else None,
                'duration_seconds': duration}
