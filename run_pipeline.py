from pathlib import Path
import sys

def main():
    CODE_DIR = Path("/home/user/Scrivania/Videos/code-analysis/package_GV/video_test").resolve()
    #VIDEO_DIR = Path("/home/user/Scrivania/Videos/December2023/Quantitative/forTraj").resolve()
    TRAJ_DIR = Path("/home/user/Scrivania/Videos/code-analysis/package_GV/video_test").resolve()
    SETTINGS_TXT = Path("/home/user/Scrivania/Videos/code-analysis/package_GV/video_test/Clamidomoni_settings.txt").resolve()
    ILLUM_DIR = Path("/home/user/Scrivania/Videos/December2023/Quantitative/forTraj").resolve()
    RADIAL_DIR = Path("/home/user/Scrivania/Videos/inputSimulations").resolve()
    RESULTS_DIR = Path("/home/user/Scrivania/Videos/code-analysis/package_GV/video_test/results").resolve()

    # Qui vanno i codici, non i video
    sys.path.insert(0, str(CODE_DIR))

    from clam_pipeline.cli import main as pipeline_main

    pipeline_main(
        code_dir=CODE_DIR,
        #video_dir=VIDEO_DIR,
        traj_dir=TRAJ_DIR,
        settings_txt=SETTINGS_TXT,
        illum_dir=ILLUM_DIR,
        radial_dir=RADIAL_DIR,
        results_dir=RESULTS_DIR,
        exp_id=68,
    )

if __name__ == "__main__":
    main()