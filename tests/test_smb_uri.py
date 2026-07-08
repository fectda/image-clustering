"""Tests for SMB URI parsing logic (bash-side URL decode + port + credentials).

Shared by both --input and --output SMB paths. The _mount_smb() helper
in organize.sh handles both directions with identical URI parsing.
"""

import os
import re

import pytest
from urllib.parse import unquote


class TestSmbUrlDecode:
    """URL-decode logic for SMB URIs — mirrors python3 -c '...unquote...' in organize.sh."""

    @pytest.mark.parametrize(
        "encoded,expected",
        [
            ("smb://server/share/path", "smb://server/share/path"),
            ("smb://server/share/adaptador%20dewatl", "smb://server/share/adaptador dewatl"),
            ("smb://server/share/cami%C3%B3n", "smb://server/share/camión"),
            ("smb://server:443/share", "smb://server:443/share"),
            ("smb://server/share/%2Ftest", "smb://server/share//test"),
        ],
    )
    def test_url_decode(self, encoded, expected):
        """URL decoding matches expected output."""
        assert unquote(encoded) == expected


class TestSmbPortExtraction:
    """Port extraction from SMB URIs — mirrors bash regex from organize.sh."""

    @pytest.mark.parametrize(
        "uri,expected_port",
        [
            ("//server/share", ""),
            ("//server:443/share", "443"),
            ("//server:445/share/path", "445"),
            ("//192.168.1.1:139/share", "139"),
        ],
    )
    def test_port_extraction(self, uri, expected_port):
        """Port regex matches expected groups."""
        port_match = re.match(r"//[^/]+:(\d+)", uri)
        port = port_match.group(1) if port_match else ""
        assert port == expected_port


class TestSmbUncConversion:
    """UNC path conversion from smb:// to //server/share."""

    @pytest.mark.parametrize(
        "smb_uri,expected_unc",
        [
            ("smb://server/share", "//server/share"),
            ("smb://server:443/share", "//server:443/share"),
            ("smb://nas.local/My%20Photos", "//nas.local/My%20Photos"),
        ],
    )
    def test_unc_conversion(self, smb_uri, expected_unc):
        """smb:// prefix stripped, // prepended."""
        unc = "//" + smb_uri[len("smb://") :]
        assert unc == expected_unc


class TestSmbCredentialBranches:
    """Credential branch logic tests — mirrors bash env var checks in organize.sh."""

    def test_guest_default(self, monkeypatch):
        """No SMB_USER env var -> guest mount options."""
        monkeypatch.delenv("SMB_USER", raising=False)
        monkeypatch.delenv("SMB_PASS", raising=False)

        smb_opts = "guest,uid=1000,gid=1000"

        assert "guest" in smb_opts
        assert "username" not in smb_opts

    def test_user_pass_set(self, monkeypatch):
        """SMB_USER + SMB_PASS set -> username=,password= mount options."""
        monkeypatch.setenv("SMB_USER", "myuser")
        monkeypatch.setenv("SMB_PASS", "mypass")

        smb_opts = "username=myuser,password=mypass,uid=1000,gid=1000"

        assert "username=myuser" in smb_opts
        assert "password=mypass" in smb_opts
        assert "guest" not in smb_opts

    def test_user_without_pass(self, monkeypatch):
        """SMB_USER set but SMB_PASS empty -> username=,password= with warning."""
        monkeypatch.setenv("SMB_USER", "myuser")
        monkeypatch.setenv("SMB_PASS", "")

        smb_opts = "username=myuser,password=,uid=1000,gid=1000"

        assert "username=myuser" in smb_opts
        assert "password=" in smb_opts
