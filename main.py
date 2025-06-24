# main.py
# 역할: 프로그램을 시작하고, 예외 발생 시 안전하게 종료하는 유일한 역할.
from driver_monitor import DriverMonitor

if __name__ == '__main__':
    try:
        import mediapipe
        try:
            import mediapipe_models
            print(f"mediapipe version: {mediapipe.__version__}, mediapipe-models version: {mediapipe_models.__version__}")
        except ImportError:
            print(f"mediapipe version: {mediapipe.__version__}, mediapipe-models not installed")
    except Exception as e:
        print(f"mediapipe version check error: {e}")
    # 메인 시스템 객체를 생성합니다.
    monitor = DriverMonitor()
    try:
        # 시스템을 시작합니다.
        monitor.start()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
    finally:
        # 프로그램이 어떤 이유로든 종료될 때, 모든 리소스를 안전하게 해제합니다.
        print("Exiting program. Releasing all resources.")
        monitor.close_all() 