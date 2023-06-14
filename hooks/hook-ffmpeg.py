from PyInstaller.utils.hooks import collect_data_files

# The following line finds all the DLL files that the av package needs and includes them in the final bundle.
binaries = collect_data_files('ffmpeg')
