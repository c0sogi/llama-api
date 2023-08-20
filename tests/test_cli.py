import json
from os import environ
import unittest
from llama_api.shared.config import AppSettingsCliArgs, MainCliArgs


class TestCLIArgs(unittest.TestCase):
    def test_cli_args(self):
        parser = MainCliArgs.parser
        environ_key = "LLAMA_CLI_ARGS"
        environ_key_prefix = "LLAMA_"

        # Check that `--install-pkgs` is inherited from `MainCliArgs`
        args = parser.parse_args(["--install-pkgs", "--port", "8080"])
        AppSettingsCliArgs.load_from_namespace(args)
        self.assertFalse(AppSettingsCliArgs.force_cuda.value)
        self.assertTrue(AppSettingsCliArgs.install_pkgs.value)
        self.assertFalse(MainCliArgs.force_cuda.value)
        self.assertTrue(MainCliArgs.install_pkgs.value)
        self.assertEqual(MainCliArgs.port.value, 8000)

        # Check that both `--force-cuda` and `--port` are inherited from `MainCliArgs`  # noqa
        args = parser.parse_args(["--port", "9000", "--force-cuda"])
        MainCliArgs.load_from_namespace(args)
        self.assertTrue(AppSettingsCliArgs.force_cuda.value)
        self.assertFalse(AppSettingsCliArgs.install_pkgs.value)
        self.assertTrue(MainCliArgs.force_cuda.value)
        self.assertFalse(MainCliArgs.install_pkgs.value)
        self.assertEqual(MainCliArgs.port.value, 9000)

        # Set `--install-pkgs` to `False` and check that it is applied
        args.install_pkgs = True
        AppSettingsCliArgs.load_from_namespace(args)
        self.assertTrue(AppSettingsCliArgs.force_cuda.value)
        self.assertTrue(AppSettingsCliArgs.install_pkgs.value)
        self.assertTrue(MainCliArgs.force_cuda.value)
        self.assertTrue(MainCliArgs.install_pkgs.value)
        self.assertEqual(MainCliArgs.port.value, 9000)

        environ[environ_key] = json.dumps({"force_cuda": False, "port": 7000})
        AppSettingsCliArgs.load_from_environ(environ_key, environ_key_prefix)
        self.assertFalse(AppSettingsCliArgs.force_cuda.value)
        self.assertTrue(AppSettingsCliArgs.install_pkgs.value)
        self.assertFalse(MainCliArgs.force_cuda.value)
        self.assertTrue(MainCliArgs.install_pkgs.value)
        self.assertEqual(MainCliArgs.port.value, 9000)

        MainCliArgs.load_from_environ(environ_key, environ_key_prefix)
        self.assertFalse(AppSettingsCliArgs.force_cuda.value)
        self.assertTrue(AppSettingsCliArgs.install_pkgs.value)
        self.assertFalse(MainCliArgs.force_cuda.value)
        self.assertTrue(MainCliArgs.install_pkgs.value)
        self.assertEqual(MainCliArgs.port.value, 7000)

        environ[f"{environ_key_prefix}MAX_SEMAPHORES"] = "100"
        MainCliArgs.load_from_environ(environ_key, environ_key_prefix)
        self.assertEqual(MainCliArgs.max_semaphores.value, 100)


if __name__ == "__main__":
    unittest.main()
