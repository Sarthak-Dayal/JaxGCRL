{
  description = "JaxGCRL Nix Flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        baseTargetPkgs =
          pkgs: with pkgs; [
            # basic shell + core tools
            bashInteractive
            coreutils
            findutils
            gawk
            gnugrep
            gnutar
            gzip
            gnused
            procps
            util-linux

    	    binutils
    	    glibc
    	    glibc.dev
    	    stdenv.cc.cc
    	    cmake
    	    gnumake

            # dev-ish stuff
            gcc
            gdb
            git
            pkg-config

            # networking-ish tools
            iproute2
            inetutils

            # runtime libs for Python wheels
            glib
	    cacert
	    zlib
            bzip2
            xz
            openssl
            libffi
            libuuid

            # python tooling
            uv
            python312
	    python313
            python314

            # --- OpenGL + X11 stack for MuJoCo/GLFW ---
            mesa # provides libGL, libEGL, etc.
            libGL
            libGLU

            xorg.libX11
            xorg.libXext
            xorg.libXrandr
            xorg.libXcursor
            xorg.libXtst
	    xorg.libXi
            xorg.libXinerama
            xorg.libXrender
            xorg.libxcb

	    freetype
	    fontconfig

            glfw # GLFW backend for MUJOCO_GL=glfw

	    cudaPackages.nsight_systems
	    cudaPackages.nsight_compute

          ];

        extraMountsCommon = [
          {
            source = "/dev";
            target = "/dev";
            recursive = true;
          }
          {
            source = "/sys";
            target = "/sys";
            recursive = true;
          }
          {
            source = "/run";
            target = "/run";
            recursive = true;
          }
          {
            source = "/etc/ssl";
            target = "/etc/ssl";
            recursive = true;
          }
          # Allow X11 apps (GLFW) inside FHS env to talk to your host X server
          {
            source = "/tmp/.X11-unix";
            target = "/tmp/.X11-unix";
            recursive = true;
          }
        ];
      in
      {
        packages.fhs-dev = pkgs.buildFHSEnv {
          name = "fhs-ubuntu-dev";
          targetPkgs = baseTargetPkgs;
          extraOutputsToInstall = [
            "dev"
            "out"
          ];
          extraMounts = extraMountsCommon;

          # This is what runs INSIDE the FHS env when devshell starts
          runScript = pkgs.writeShellScript "dev-entry" ''
            set -euo pipefail
            echo "[dev] cwd at startup: $PWD"

            # Ensure venv exists (only sync if .venv missing)
            echo "[dev] running: uv sync --python 3.13"
	          uv sync --python 3.13 --extra cuda13

            # "Activate" the venv by mutating PATH + VIRTUAL_ENV
            if [ -d ".venv/bin" ]; then
              export VIRTUAL_ENV="$PWD/.venv"
              export PATH="$VIRTUAL_ENV/bin:$PATH"
              echo "[dev] using venv at $VIRTUAL_ENV"
            else
              echo "[dev] WARNING: .venv/bin missing; continuing without venv"
            fi

            echo "[dev] dropping into interactive bash"
            bash
          '';
        };

	devShells.default = pkgs.mkShell {
          buildInputs = [ self.packages.${system}.fhs-dev ];

          # Immediately jump into the FHS dev env when you `nix develop`
          shellHook = ''
            echo "[devShell] entering FHS dev environment..."
    	    export EXTRA_CCFLAGS="-I/usr/include"
	    export SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt

	    export CC=${pkgs.gcc}/bin/gcc
            export CXX=${pkgs.gcc}/bin/g++
            export LD=${pkgs.binutils}/bin/ld

            exec ${self.packages.${system}.fhs-dev}/bin/fhs-ubuntu-dev
          '';
        };
      }
    );
}
