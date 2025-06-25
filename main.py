# main.py
# 역할: 프로그램을 시작하고, 예외 발생 시 안전하게 종료하는 유일한 역할.
from driver_monitor import DriverMonitor
import config as cfg
import traceback

def main():
    """
    애플리케이션의 메인 진입점.
    DriverMonitor 객체를 생성하고 실행합니다.
    """
    monitor = None
    try:
        monitor = DriverMonitor(camera_id=cfg.VIDEO_PATH)
        monitor.run()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
    finally:
        if monitor:
            monitor.close()
        print("Application terminated.")

if __name__ == '__main__':
    main() 