import ast
import subprocess
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


def _keyword_constant(call, name):
    for keyword in call.keywords:
        if keyword.arg == name and isinstance(keyword.value, ast.Constant):
            return keyword.value.value
    return None


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
        },
    )


def test_shared_emt_quark_and_gluon_flow_defaults_match():
    quark = _source(
        "application/EMT_disconnected_1pt/perlmutter/"
        "Pyquda_EMT_disconnected_quark_1pt.py"
    )
    gluon = _source(
        "application/EMT_disconnected_1pt/perlmutter/"
        "Pyquda_EMT_disconnected_gluon_1pt.py"
    )
    expected = 'os.environ.get("EMT_1PT_FLOW_EPSILON", "0.207936")'
    assert expected in quark
    assert expected in gluon
    assert "EMT_1PT_SRC_POS" not in gluon
    assert "EMT_1PT_SRC_T" not in gluon


def test_emt_quark_hp_defaults_to_isotropic_four_dimensional_ordering():
    quark = _source(
        "application/EMT_disconnected_1pt/perlmutter/"
        "Pyquda_EMT_disconnected_quark_1pt.py"
    )
    wrapper = _source(
        "application/EMT_disconnected_1pt/perlmutter/run_quark_1pt.sh"
    )
    ordering = "interleaved_xyzt_binary_projected_to_evenodd"
    assert f'os.environ.get("EMT_1PT_HP_ORDERING", "{ordering}")' in quark
    assert f'EMT_1PT_HP_ORDERING:-{ordering}' in wrapper


def test_emt_quark_flow_batch_size_is_positive_cli_only():
    driver = (
        "application/EMT_disconnected_1pt/perlmutter/"
        "Pyquda_EMT_disconnected_quark_1pt.py"
    )
    wrapper = "application/EMT_disconnected_1pt/perlmutter/run_quark_1pt.sh"
    argument = _parser_arguments(driver)["--flow-batch-size"]
    assert _keyword_constant(argument, "default") == 1
    assert "flow_batch_size=args.flow_batch_size" in _source(driver)
    assert "EMT_1PT_FLOW_BATCH_SIZE" not in _source(driver)
    assert "EMT_1PT_FLOW_BATCH_SIZE" not in _source(wrapper)

    for invalid in ("0", "-1", "1.5", "bad"):
        result = subprocess.run(
            [
                "bash", str(REPO_ROOT / wrapper),
                "--config_num", "0", "--flow-batch-size", invalid,
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2
        assert "positive integer" in result.stderr


def test_disconnected_configuration_is_required_cli_only():
    single_config_scripts = {
        "application/EMT_disconnected_1pt/perlmutter/Pyquda_EMT_disconnected_quark_1pt.py": "EMT_1PT_CONFIG_NUM",
        "application/EMT_disconnected_1pt/perlmutter/Pyquda_EMT_disconnected_gluon_1pt.py": "EMT_1PT_CONFIG_NUM",
        "application/EMT_disconnected_1pt/perlmutter/Pyquda_EMT_disconnected_proton_2pt.py": "EMT_1PT_CONFIG_NUM",
        "application/EMT_disconnected_1pt/perlmutter/Pyquda_EMT_disconnected_finalize_quark_1pt.py": "EMT_1PT_CONFIG_NUM",
        "application/qTMD_disconnected_1pt/perlmutter/Pyquda_Disconnected_qTMD_1pt.py": "QTMD_1PT_CONFIG_NUM",
        "application/qTMD_disconnected_1pt/perlmutter/Pyquda_Disconnected_qTMD_finalize_1pt.py": "QTMD_1PT_CONFIG_NUM",
        "application/flowed_quark_ringed_norm/perlmutter/Pyquda_flowed_quark_ringed_norm.py": "FLOWED_RINGED_CONFIG_NUM",
        "application/flowed_quark_ringed_norm/Aurora/Pyquda_flowed_quark_ringed_norm.py": "FLOWED_RINGED_CONFIG_NUM",
    }
    for relpath, removed_env in single_config_scripts.items():
        arguments = _parser_arguments(relpath)
        assert _keyword_constant(arguments["--config_num"], "required") is True
        assert removed_env not in _source(relpath)
        assert "conf = args.config_num" in _source(relpath)

    build = (
        "application/EMT_disconnected_1pt/perlmutter/"
        "Pyquda_EMT_disconnected_build_3pt.py"
    )
    arguments = _parser_arguments(build)
    assert "--config_num" not in arguments
    assert _keyword_constant(arguments["--configs"], "required") is True
    assert "configs = parse_int_list(args.configs)" in _source(build)
    assert "EMT_1PT_CONFIG_NUM" not in _source(build)
    assert "EMT_DISC_CONFIGS" not in _source(build)


def test_disconnected_builder_is_quark_only_by_default():
    build = (
        "application/EMT_disconnected_1pt/perlmutter/"
        "Pyquda_EMT_disconnected_build_3pt.py"
    )
    arguments = _parser_arguments(build)
    include_gluon = arguments["--include_gluon"]
    assert _keyword_constant(include_gluon, "action") == "store_true"
    source = _source(build)
    assert "if args.include_gluon" in source
    assert 'h5.attrs["includes_gluon"] = bool(args.include_gluon)' in source

    wrapper = _source(
        "application/EMT_disconnected_1pt/perlmutter/"
        "run_build_disconnected_3pt.sh"
    )
    assert "[--include_gluon]" in wrapper


def test_connected_emt_configuration_is_required_cli_only():
    scripts = [
        "application/EMT_meson/perlmutter/Pyquda_EMT_quark_3pt.py",
        "application/EMT_proton/perlmutter/Pyquda_EMT_proton_quark_3pt.py",
        "application/EMT_proton/Aurora/Pyquda_EMT_proton_quark_3pt.py",
    ]
    for relpath in scripts:
        arguments = _parser_arguments(relpath)
        assert _keyword_constant(arguments["--config_num"], "required") is True
        source = _source(relpath)
        assert "parse_known_args" not in source
        assert "CONFIG_NUM" not in source


def test_emt_removed_entrypoints_and_aliases_stay_removed():
    removed = [
        "application/EMT_meson/frontier/Pyquda_EMT_quark_3pt.py",
        "application/EMT_meson/frontier/submit_quark_3pt.sh",
        "application/EMT_proton/perlmutter/Pyquda_EMT_proton_quark_1pt.py",
        "application/EMT_proton/perlmutter/run_proton_quark_1pt.sh",
        "application/EMT_proton/perlmutter/submit_proton_quark_1pt.sh",
    ]
    assert all(not (REPO_ROOT / relpath).exists() for relpath in removed)
    assert "class GluonEMT" not in _source("pyquda_measurement_utils/pion_EMT_vibe_develop.py")
    assert "class ProtonGluonEMT" not in _source("pyquda_measurement_utils/proton_EMT_vibe_develop.py")


def test_active_perlmutter_emt_defaults_use_current_software_tree():
    paths = list((REPO_ROOT / "application").glob("EMT_*/perlmutter/*"))
    paths.append(REPO_ROOT / "systems/perlmutter/activate-venv-quda.sh")
    for path in paths:
        if path.is_file() and path.suffix in {".py", ".sh", ""}:
            source = path.read_text()
            assert "/global/cfs/cdirs/m3760" not in source
    activation = _source("systems/perlmutter/activate-venv-quda.sh")
    assert "venv-quda-develop" in activation
    assert "quda-develop/install" in activation


def test_pion_connected_driver_has_no_shared_output_override():
    source = _source("application/EMT_meson/perlmutter/Pyquda_EMT_quark_3pt.py")
    assert "EMT_3PT_OUT" not in source
    assert "EMT_2PT_OUT" not in source
    assert "get_emt_quark_3pt_file_tag" in source
    assert "spin=5" not in source
    assert ".spin" not in source
    assert "src_interpolator" in source
    assert "sink_interpolator" in source
    assert '"--pos-boost"' in source
    assert '"--neg-boost"' in source
    assert '"pos_boost": args.pos_boost' in source
    assert '"neg_boost": args.neg_boost' in source
    assert "default_sm_tag" in source
    assert "boost_tag(args.pos_boost)" in source
    assert "boost_tag(args.neg_boost)" in source
    assert 'os.environ.get("EMT_LAT_TAG", "S8T32")' in source
    assert "1e-10" in source


def test_pion_connected_wrappers_forward_boost_cli_and_reject_bad_values():
    wrappers = [
        "application/EMT_meson/perlmutter/run_quark_3pt.sh",
        "application/EMT_meson/perlmutter/submit_quark_3pt.sh",
    ]
    for relpath in wrappers:
        source = _source(relpath)
        assert "--pos-boost" in source
        assert "--neg-boost" in source
        result = subprocess.run(
            [
                "bash", str(REPO_ROOT / relpath), "--config_num", "1",
                "--pos-boost", "0.1", "--neg-boost", "0.0.-1",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2
        assert "--pos-boost" in result.stderr


def test_emt_backend_transfers_do_not_call_cupy_get_directly():
    pion = _source("pyquda_measurement_utils/pion_EMT_vibe_develop.py")
    gluon = _source(
        "pyquda_measurement_utils/Disconnected_1pt_EMT_vibe_develop.py"
    )
    assert 'contract("qwtzyx, gwtzyx -> gqt", phases_2pt, scalar).get()' not in pion
    assert 'contract("qwtzyx, wtzyx -> qt", phases_3pt, tmp).get()' not in gluon
    assert "array_to_numpy" in pion
    assert "array_to_numpy" in gluon


def test_proton_emt_parameters_use_only_canonical_boost_names():
    module = _source("pyquda_measurement_utils/proton_EMT_vibe_develop.py")
    assert "self.save_propagators" not in module
    assert "self.pos_boost" not in module
    assert "self.neg_boost" not in module
    assert 'parameters["boost_in"]' in module
    assert 'parameters["boost_out"]' in module


def test_proton_qtmd_parameters_use_only_active_measurement_fields():
    module = _source("pyquda_measurement_utils/proton_qTMD_pyquda.py")
    assert "self.save_propagators" not in module
    assert "self.pos_boost" not in module
    assert "self.boost_out" not in module
    assert 'parameters["boost_in"]' in module

    for relpath in [
        "application/nucleon_TMD/perlmutter/Pyquda_nucleon_TMD.py",
        "application/nucleon_TMD/Aurora/pyquda_nucleon_TMD_GI.py",
        "application/nucleon_TMD_CG/Aurora/pyquda_nucleon_TMD.py",
    ]:
        source = _source(relpath)
        assert '"save_propagators"' not in source
        assert "my_pyquda_gammas" not in source
        assert "gamma_stack" in source

    driver = _source(
        "application/nucleon_TMD/perlmutter/Pyquda_nucleon_TMD.py"
    )
    assert driver.count("if latt_info.mpi_rank != 0:\n        return") >= 2
    assert 'for flavor in ("D", "U")' in driver


def test_pion_channel_provenance_is_explicit_and_rank_zero_writes_qtmd():
    for relpath in ["application/pion_TMD/perlmutter/Pyquda_pion_TMD.py"]:
        source = _source(relpath)
        assert "get_pion_channel_tag" in source
        assert '"src_interpolator"' in source
        assert '"sink_interpolator"' in source
        assert '"operator_gamma"' in source
        assert "source_gamma_provenance" in source
        assert "tasks if rank == 0 else ()" in source
        assert '"pos_boost"' in source
        assert '"neg_boost"' in source
        assert '"operator_insertion_line": "neg_boost"' in source
        assert '"boost_line_convention": "pos_spectator_neg_active"' in source
        assert "parse_known_args" not in source

    assert not (REPO_ROOT / "application/pion_TMD_CG").exists()

    emff = _source("application/EMFF_pion/perlmutter/Pyquda_pion_EMFF.py")
    assert "get_pion_channel_tag" in emff
    assert '"src_interpolator"' in emff
    assert '"sink_interpolator"' in emff
    assert '"current_gamma_basis": "all_16"' in emff
    assert "source_gamma_provenance" in emff

    for relpath in [
        "application/EMFF_pion_background_response/perlmutter/Pyquda_pion_EMFF_background_response.py",
        "application/pion_current_current_response/perlmutter/Pyquda_pion_current_current_response.py",
    ]:
        source = _source(relpath)
        assert "source_gamma_provenance" in source
        assert ".src{src_gamma}" in source
    assert "channel_set_tag" in emff


def test_standalone_ringed_exposes_cli_only_flow_batch_size():
    drivers = [
        "application/flowed_quark_ringed_norm/perlmutter/Pyquda_flowed_quark_ringed_norm.py",
        "application/flowed_quark_ringed_norm/Aurora/Pyquda_flowed_quark_ringed_norm.py",
    ]
    for relpath in drivers:
        argument = _parser_arguments(relpath)["--flow-batch-size"]
        assert _keyword_constant(argument, "default") == 1
        source = _source(relpath)
        assert "flow_batch_size=args.flow_batch_size" in source
        assert "FLOWED_RINGED_FLOW_BATCH" not in source

    wrappers = [
        "application/flowed_quark_ringed_norm/perlmutter/run_flowed_quark_ringed_norm.sh",
        "application/flowed_quark_ringed_norm/perlmutter/run_login_smoke.sh",
        "application/flowed_quark_ringed_norm/Aurora/run_flowed_quark_ringed_norm.sh",
        "application/flowed_quark_ringed_norm/Aurora/submit_or_run_interactive.sh",
    ]
    for relpath in wrappers:
        source = _source(relpath)
        assert "--flow-batch-size" in source
        assert "FLOWED_RINGED_FLOW_BATCH" not in source


def test_connected_emt_shell_wrappers_require_named_configuration():
    wrappers = [
        "application/EMT_meson/perlmutter/run_quark_3pt.sh",
        "application/EMT_meson/perlmutter/submit_quark_3pt.sh",
        "application/EMT_proton/perlmutter/run_proton_quark_3pt.sh",
        "application/EMT_proton/perlmutter/submit_proton_quark_3pt.sh",
        "application/EMT_proton/Aurora/run_proton_quark_3pt.sh",
        "application/EMT_proton/Aurora/submit_or_run_interactive.sh",
    ]
    for relpath in wrappers:
        result = subprocess.run(
            ["bash", str(REPO_ROOT / relpath)], capture_output=True, text=True
        )
        assert result.returncode == 2
        assert "--config_num" in result.stderr


def test_connected_emt_entrypoints_share_cli_multigrid_default():
    drivers = [
        "application/EMT_meson/perlmutter/Pyquda_EMT_quark_3pt.py",
        "application/EMT_proton/perlmutter/Pyquda_EMT_proton_quark_3pt.py",
        "application/EMT_proton/Aurora/Pyquda_EMT_proton_quark_3pt.py",
        "application/EMT_disconnected_1pt/perlmutter/Pyquda_EMT_disconnected_proton_2pt.py",
    ]
    for relpath in drivers:
        arguments = _parser_arguments(relpath)
        assert _keyword_constant(arguments["--mg-block"], "default") == "8.8.4.4"
        source = _source(relpath)
        assert "parse_optional_multigrid_blocks(args.mg_block)" in source
        assert "EMT_PROTON_MG_BLOCK" not in source

    for relpath in (
        "pyquda_measurement_utils/pion_EMT_vibe_develop.py",
        "pyquda_measurement_utils/proton_EMT_vibe_develop.py",
    ):
        source = _source(relpath)
        assert "self.multigrid_blocks" in source


def test_disconnected_shell_wrappers_reject_missing_or_unknown_configuration():
    wrappers = [
        "application/EMT_disconnected_1pt/perlmutter/run_quark_1pt.sh",
        "application/EMT_disconnected_1pt/perlmutter/run_gluon_1pt.sh",
        "application/EMT_disconnected_1pt/perlmutter/run_proton_2pt.sh",
        "application/EMT_disconnected_1pt/perlmutter/run_finalize_quark_1pt.sh",
        "application/qTMD_disconnected_1pt/perlmutter/run_qTMD_1pt.sh",
        "application/qTMD_disconnected_1pt/perlmutter/run_finalize_qTMD_1pt.sh",
        "application/qTMD_disconnected_1pt/perlmutter/submit_qTMD_1pt.sh",
        "application/flowed_quark_ringed_norm/perlmutter/run_flowed_quark_ringed_norm.sh",
        "application/flowed_quark_ringed_norm/perlmutter/run_login_smoke.sh",
        "application/flowed_quark_ringed_norm/Aurora/run_flowed_quark_ringed_norm.sh",
        "application/flowed_quark_ringed_norm/Aurora/submit_or_run_interactive.sh",
    ]
    removed_envs = {
        "EMT_1PT_CONFIG_NUM", "QTMD_1PT_CONFIG_NUM", "FLOWED_RINGED_CONFIG_NUM"
    }
    for relpath in wrappers:
        source = _source(relpath)
        for removed_env in removed_envs:
            assert removed_env not in source
        missing = subprocess.run(
            ["bash", str(REPO_ROOT / relpath)], capture_output=True, text=True
        )
        unknown = subprocess.run(
            ["bash", str(REPO_ROOT / relpath), "--unknown", "7"],
            capture_output=True,
            text=True,
        )
        assert missing.returncode == 2
        assert unknown.returncode == 2

    build = "application/EMT_disconnected_1pt/perlmutter/run_build_disconnected_3pt.sh"
    source = _source(build)
    assert "EMT_1PT_CONFIG_NUM" not in source
    assert "EMT_DISC_CONFIGS" not in source
    for args in ([], ["--config_num", "7"], ["--configs", ""], ["--configs", "7,"]):
        result = subprocess.run(
            ["bash", str(REPO_ROOT / build), *args], capture_output=True, text=True
        )
        assert result.returncode == 2


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
