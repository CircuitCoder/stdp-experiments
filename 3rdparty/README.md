# Reviewed third-party sources

These are shallow research clones used to audit current STDP implementations.
They are intentionally separate from the experiment repository history. The
simulator also uses the nlohmann JSON headers from the local `json` clone.

| Directory | Upstream | Reviewed commit |
|---|---|---|
| `genn` | <https://github.com/genn-team/genn> | `563c45c531eb6adce53ad3ff3f46d614a19abdb2` |
| `brian2` | <https://github.com/brian-team/brian2> | `1bfa1a9275bd9672b49f4bf61ffbaf6f7cb55fc9` |
| `brian2cuda` | <https://github.com/brian-team/brian2cuda> | `825c0c58d2a0b2bf471af7fc97e184e724522845` |
| `nest-simulator` | <https://github.com/nest/nest-simulator> | `182eba446a8b89108f21cd2ad54aa4c667afd86a` |
| `nest-gpu` | <https://github.com/nest/nest-gpu> | `830b15ba1d9204346cd5e83eef21a96018daac69` |
| `CARLsim6` | <https://github.com/UCI-CARL/CARLsim6> | `d527c55afba76f488053fb4c36f6adeebd01a5fa` |
| `json` | <https://github.com/nlohmann/json> | `3565f40229515411177c196ec912a06307802ed6` |

Recreate a clone with `git clone --depth 1 URL 3rdparty/DIRECTORY` and verify
that `git -C 3rdparty/DIRECTORY rev-parse HEAD` matches this table. If upstream
has advanced, fetch and check out the recorded commit explicitly.
