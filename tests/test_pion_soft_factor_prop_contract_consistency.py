def _mom_tag(momentum):
    return tuple(int(v) for v in momentum)


def _required_contract_momenta(quark_mom, tsep_list, t_start, t_count, Lt):
    required = set()
    tslice_list = [(t_start + dt) % Lt for dt in range(t_count)]
    for tslice in tslice_list:
        pos_t = tslice
        for quark_mom_fw in quark_mom:
            for quark_mom_bw in quark_mom:
                required.add((pos_t, _mom_tag(quark_mom_fw)))
                required.add((pos_t, _mom_tag([-quark_mom_bw[0], -quark_mom_bw[1], -quark_mom_bw[2]])))
                for tsep in tsep_list:
                    sink_t = (tslice + tsep) % Lt
                    required.add((sink_t, _mom_tag([-quark_mom_fw[0], -quark_mom_fw[1], -quark_mom_fw[2]])))
                    required.add((sink_t, _mom_tag(quark_mom_bw)))
    return required


def _available_prop_momenta(quark_mom, t_start, t_count, Lt):
    momenta = []
    for mom in quark_mom:
        momenta.append(mom)
        momenta.append([-mom[0], -mom[1], -mom[2]])
    momenta = [list(mom) for mom in dict.fromkeys(tuple(mom) for mom in momenta)]
    tslice_list = [(t_start + dt) % Lt for dt in range(t_count)]
    return {(tslice, _mom_tag(mom)) for tslice in tslice_list for mom in momenta}


def _prop_stage_momenta_to_save(quark_mom):
    momenta = []
    for mom in quark_mom:
        momenta.append(mom)
        momenta.append([-mom[0], -mom[1], -mom[2]])
    return [list(mom) for mom in dict.fromkeys(tuple(mom) for mom in momenta)]


def test_soft_factor_prop_stage_saves_all_contract_required_momenta_when_all_times_are_generated():
    quark_mom = [[0, 0, 1], [0, 0, 2]]
    Lt = 8
    required = _required_contract_momenta(quark_mom, [2, 4], 6, Lt, Lt)
    available = _available_prop_momenta(quark_mom, 0, Lt, Lt)

    assert required <= available


def test_soft_factor_limited_prop_generation_must_cover_sink_times_for_tsep_list():
    quark_mom = [[0, 0, 1]]
    Lt = 8
    required = _required_contract_momenta(quark_mom, [2, 4], 0, 1, Lt)
    available = _available_prop_momenta(quark_mom, 0, 1, Lt)

    assert required - available == {
        (2, _mom_tag([0, 0, -1])),
        (2, _mom_tag([0, 0, 1])),
        (4, _mom_tag([0, 0, -1])),
        (4, _mom_tag([0, 0, 1])),
    }


def test_soft_factor_prop_application_deduplicates_plus_minus_momenta():
    momenta_to_save = _prop_stage_momenta_to_save([[0, 0, 4], [0, 0, 5]])
    assert [_mom_tag(mom) for mom in momenta_to_save] == [(0, 0, 4), (0, 0, -4), (0, 0, 5), (0, 0, -5)]
