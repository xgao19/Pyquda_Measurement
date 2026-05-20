import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _source(relpath):
    return (REPO_ROOT / relpath).read_text()


def _tree(relpath):
    return ast.parse(_source(relpath))


def _parser_arguments(relpath):
    arguments = {}
    for node in ast.walk(_tree(relpath)):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        name = node.args[0].value
        if not isinstance(name, str) or not name.startswith("--"):
            continue
        arguments[name] = node
    return arguments


def _env_names(node):
    names = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if not isinstance(child.func, ast.Attribute) or child.func.attr != "get":
            continue
        value = child.func.value
        if not (
            isinstance(value, ast.Attribute)
            and value.attr == "environ"
            and isinstance(value.value, ast.Name)
            and value.value.id == "os"
        ):
            continue
        if child.args and isinstance(child.args[0], ast.Constant):
            names.add(child.args[0].value)
    return names


def _exported_env_names(relpath):
    names = set()
    for line in _source(relpath).splitlines():
        stripped = line.strip()
        if not stripped.startswith("export "):
            continue
        assignment = stripped[len("export ") :]
        if "=" in assignment:
            names.add(assignment.split("=", 1)[0])
    return names


def _assert_parser_envs(relpath, expected):
    arguments = _parser_arguments(relpath)
    assert set(expected) <= set(arguments)
    for arg_name, env_name in expected.items():
        assert env_name in _env_names(arguments[arg_name]), f"{relpath} {arg_name} does not read {env_name}"


def _literal_dict_assignment(relpath, name):
    for node in _tree(relpath).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return node.value
    raise AssertionError(f"Could not find assignment {name} in {relpath}")


def _dict_keys(dict_node):
    assert isinstance(dict_node, ast.Dict)
    keys = []
    for key in dict_node.keys:
        assert isinstance(key, ast.Constant)
        keys.append(key.value)
    return set(keys)


def test_connected_tmd_application_env_argument_mapping():
    _assert_parser_envs(
        "application/pion_TMD/perlmutter/Pyquda_pion_TMD.py",
        {
            "--config_num": "PION_TMD_CONFIG_NUM",
            "--mpi_geometry": "PION_TMD_MPI_GEOMETRY",
            "--data_dir": "PION_TMD_DATA_DIR",
            "--qmax": "PION_TMD_QMAX",
            "--b_z": "PION_TMD_BZ",
            "--b_T": "PION_TMD_BT",
            "--eta": "PION_TMD_ETA",
            "--run_cg_qtmd": "PION_TMD_RUN_CG_QTMD",
            "--run_gi_qtmd": "PION_TMD_RUN_GI_QTMD",
            "--run_pdf": "PION_TMD_RUN_PDF",
            "--gi_staple_mode": "PION_TMD_GI_STAPLE_MODE",
        },
    )
    _assert_parser_envs(
        "application/nucleon_TMD/perlmutter/Pyquda_nucleon_TMD.py",
        {
            "--config_num": "NUCLEON_TMD_CONFIG_NUM",
            "--mpi_geometry": "NUCLEON_TMD_MPI_GEOMETRY",
            "--data_dir": "NUCLEON_TMD_DATA_DIR",
            "--qmax": "NUCLEON_TMD_QMAX",
            "--b_z": "NUCLEON_TMD_BZ",
            "--b_T": "NUCLEON_TMD_BT",
            "--eta": "NUCLEON_TMD_ETA",
            "--run_cg_qtmd": "NUCLEON_TMD_RUN_CG_QTMD",
            "--run_gi_qtmd": "NUCLEON_TMD_RUN_GI_QTMD",
            "--run_pdf": "NUCLEON_TMD_RUN_PDF",
            "--gi_staple_mode": "NUCLEON_TMD_GI_STAPLE_MODE",
        },
    )


def test_connected_tmd_parameter_dict_contains_analysis_knobs():
    pion_keys = _dict_keys(_literal_dict_assignment("application/pion_TMD/perlmutter/Pyquda_pion_TMD.py", "parameters"))
    nucleon_keys = _dict_keys(_literal_dict_assignment("application/nucleon_TMD/perlmutter/Pyquda_nucleon_TMD.py", "parameters"))

    for keys in (pion_keys, nucleon_keys):
        assert {"eta", "b_z", "b_T", "qext", "qext_PDF", "pf", "p_2pt", "width", "t_insert"} <= keys
    assert {"pos_boost", "neg_boost"} <= pion_keys
    assert {"boost_in", "boost_out", "pol"} <= nucleon_keys


def test_pion_emff_application_env_argument_mapping_and_parameter_keys():
    _assert_parser_envs(
        "application/EMFF_pion/perlmutter/Pyquda_pion_EMFF.py",
        {
            "--config_num": "PION_EMFF_CONFIG_NUM",
            "--mpi_geometry": "PION_EMFF_MPI_GEOMETRY",
            "--qmax": "PION_EMFF_QMAX",
            "--pf": "PION_EMFF_PF",
            "--t_insert": "PION_EMFF_T_INSERT",
            "--pos_boost_src": "PION_EMFF_POS_BOOST_SRC",
            "--pos_boost_sink": "PION_EMFF_POS_BOOST_SINK",
            "--neg_boost_src": "PION_EMFF_NEG_BOOST_SRC",
            "--neg_boost_sink": "PION_EMFF_NEG_BOOST_SINK",
            "--src_interpolators": "PION_EMFF_SRC_INTERPOLATORS",
            "--sink_interpolator": "PION_EMFF_SINK_INTERPOLATOR",
        },
    )
    keys = _dict_keys(_literal_dict_assignment("application/EMFF_pion/perlmutter/Pyquda_pion_EMFF.py", "parameters"))
    assert {"qext", "pf", "p_2pt", "pos_boost_src", "pos_boost_sink", "neg_boost_src", "neg_boost_sink", "t_insert"} <= keys


def test_pion_soft_factor_application_env_argument_mapping_and_wrappers():
    prop_script = "application/pion_soft_factor/perlmutter/Pyquda_pion_soft_factor_prop.py"
    contract_script = "application/pion_soft_factor/perlmutter/Pyquda_pion_soft_factor_contract.py"
    _assert_parser_envs(
        prop_script,
        {
            "--config_num": "PION_SOFT_CONFIG_NUM",
            "--mpi_geometry": "PION_SOFT_MPI_GEOMETRY",
            "--t_start": "PION_SOFT_T_START",
            "--t_count": "PION_SOFT_T_COUNT",
            "--quark_mom_z": "PION_SOFT_QUARK_MOM_Z",
            "--do_gauge_fix": "PION_SOFT_DO_GAUGE_FIX",
        },
    )
    _assert_parser_envs(
        contract_script,
        {
            "--config_num": "PION_SOFT_CONFIG_NUM",
            "--mpi_geometry": "PION_SOFT_MPI_GEOMETRY",
            "--t_start": "PION_SOFT_T_START",
            "--t_count": "PION_SOFT_T_COUNT",
            "--quark_mom_z": "PION_SOFT_QUARK_MOM_Z",
            "--bT_dir": "PION_SOFT_BT_DIR",
            "--bT_length": "PION_SOFT_BT_LENGTH",
            "--bz_length": "PION_SOFT_BZ_LENGTH",
            "--tsep_list": "PION_SOFT_TSEP_LIST",
        },
    )

    prop_exports = _exported_env_names("application/pion_soft_factor/perlmutter/run_pion_soft_factor_prop.sh")
    contract_exports = _exported_env_names("application/pion_soft_factor/perlmutter/run_pion_soft_factor_contract.sh")
    assert {"PION_SOFT_T_COUNT", "PION_SOFT_QUARK_MOM_Z", "PION_SOFT_MASS", "PION_SOFT_CSW"} <= prop_exports
    assert {"PION_SOFT_T_COUNT", "PION_SOFT_QUARK_MOM_Z", "PION_SOFT_BT_DIR", "PION_SOFT_BT_LENGTH", "PION_SOFT_TSEP_LIST"} <= contract_exports


def test_pion_soft_factor_prop_wrapper_defaults_to_all_time_slices():
    text = _source("application/pion_soft_factor/perlmutter/run_pion_soft_factor_prop.sh")
    assert 'export PION_SOFT_T_COUNT="${PION_SOFT_T_COUNT:-0}"' in text


def test_pion_soft_factor_parameters_have_matching_prop_and_contract_core_keys():
    prop_keys = _dict_keys(_literal_dict_assignment("application/pion_soft_factor/perlmutter/Pyquda_pion_soft_factor_prop.py", "parameters"))
    contract_keys = _dict_keys(_literal_dict_assignment("application/pion_soft_factor/perlmutter/Pyquda_pion_soft_factor_contract.py", "parameters"))

    expected = {"quark_mom", "bT_dir", "bT_length", "bz_length", "tsep_list"}
    assert expected <= prop_keys
    assert expected <= contract_keys
