from pathlib import Path


def test_contact_gate_rejects_no_contact_even_with_zero_force_peaks():
    assert Path('scripts/rudra_admittance_check.py').exists()
    from scripts.rudra_admittance_check import contact_report
    rows = [dict(t=i*.01,phase='contact',contact=False,force=0.,raw_peak=0.,z=0.,xy_error=0.,yaw_error=0.,tilt=0.) for i in range(200)]
    assert not contact_report(rows)['passed']


def test_contact_gate_accepts_stable_light_contact_but_rejects_overforce():
    assert Path('scripts/rudra_admittance_check.py').exists()
    from scripts.rudra_admittance_check import contact_report
    rows = [dict(t=i*.01,phase='contact',contact=True,force=.1,raw_peak=.12,z=0.,xy_error=0.,yaw_error=0.,tilt=0.) for i in range(200)]
    assert contact_report(rows)['passed']
    rows[50]['raw_peak'] = 21.
    assert not contact_report(rows)['passed']


def test_contact_gate_rejects_wrong_brush_attitude():
    from scripts.rudra_admittance_check import contact_report
    rows = [dict(t=i*.01,phase='contact',contact=True,force=.1,raw_peak=.12,z=0.,xy_error=0.,yaw_error=.3,tilt=.1) for i in range(200)]
    assert not contact_report(rows)['passed']


def test_contact_report_distinguishes_interval_coverage_from_physics_occupancy():
    from scripts.rudra_admittance_check import contact_report
    rows = [dict(t=i*.01, phase='contact', contact=True,
                 endpoint_contact=False, physics_contact_fraction=.6,
                 force=1., raw_peak=2., z=0., xy_error=0., yaw_error=0., tilt=0.)
            for i in range(200)]
    report = contact_report(rows, target_force=1.)
    assert report['contact_fraction'] == 1.
    assert report['endpoint_contact_fraction'] == 0.
    assert abs(report['physics_contact_fraction'] - .6) < 1e-9
    assert report['contact_gate_basis'] == 'any touch within each 10ms control interval'
