from energy import PackageEnergy, counter_delta
import pytest


def test_wraparound():
    assert counter_delta(95, 5, 100) == 10
    assert counter_delta(5, 95, 100) == 90


def test_package_counted_once_and_subdomains_excluded(tmp_path):
    package = tmp_path / 'intel-rapl' / 'intel-rapl:0'
    package.mkdir(parents=True)
    for name, value in [('name', 'package-0'), ('energy_uj', '950000'),
                        ('max_energy_range_uj', '1000000')]:
        (package / name).write_text(value)
    (tmp_path / 'intel-rapl:0').symlink_to(package)
    core = package / 'intel-rapl:0:0'
    core.mkdir()
    for name, value in [('name', 'core'), ('energy_uj', '400000'),
                        ('max_energy_range_uj', '1000000')]:
        (core / name).write_text(value)
    reader = PackageEnergy(tmp_path)
    assert len(reader.domains) == 1
    reader.start()
    (package / 'energy_uj').write_text('50000')
    assert reader.stop()['joules'] == 0.1


def test_absent_counters_are_not_zero_power(tmp_path):
    reader = PackageEnergy(tmp_path)
    reader.start()
    result = reader.stop()
    assert not result['available']
    assert result['joules'] is None
    assert result['average_watts'] is None


@pytest.mark.parametrize('failure_before_start', [True, False])
def test_counter_failure_preserves_timing_without_energy(tmp_path, failure_before_start):
    package = tmp_path / 'intel-rapl:0'
    package.mkdir()
    for name, value in [('name', 'package-0'), ('energy_uj', '50'),
                        ('max_energy_range_uj', '1000')]:
        (package / name).write_text(value)
    reader = PackageEnergy(tmp_path)
    if failure_before_start:
        (package / 'energy_uj').write_text('unavailable')
    reader.start()
    if not failure_before_start:
        (package / 'energy_uj').write_text('unavailable')
    result = reader.stop()
    assert result['errors']
    assert result['duration_seconds'] >= 0
    assert result['joules'] is None
    assert result['average_watts'] is None
