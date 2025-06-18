# src/main_window.py (옵션 제어 기능 추가 최종 버전)
import cv2
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QListWidget,
    QFileDialog,
    QCheckBox,
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
from video_thread import VideoThread  # 상대 경로 import


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("운전자 상태 모니터링 시스템 (Holistic 전문가 버전)")
        self.setGeometry(100, 100, 1200, 800)

        # UI 위젯 생성
        self.video_label = QLabel("비디오/웹캠을 시작하세요", self)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet(
            "border: 2px solid black; background-color: #333;"
        )
        self.video_label.setScaledContents(True)  # 창 크기 버그 수정을 위한 설정

        self.btn_open_files = QPushButton("비디오 파일 선택", self)
        self.btn_start_webcam = QPushButton("웹캠 시작", self)
        self.btn_stop = QPushButton("중지", self)
        self.btn_stop.setEnabled(False)

        # 새로운 제어 옵션 추가
        self.check_segmentation = QCheckBox("배경 제거 활성화", self)
        self.check_flip = QCheckBox("웹캠 좌우 반전 (셀피 모드)", self)

        self.file_list_widget = QListWidget(self)
        self.status_bar = self.statusBar()

        # 레이아웃 설정
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()

        left_layout.addWidget(self.file_list_widget)
        left_layout.addWidget(self.btn_open_files)
        left_layout.addWidget(self.btn_start_webcam)
        left_layout.addWidget(self.btn_stop)
        left_layout.addSpacing(20)
        left_layout.addWidget(QLabel("--- 시각화 옵션 ---"))
        left_layout.addWidget(self.check_segmentation)
        left_layout.addWidget(self.check_flip)
        left_layout.addStretch(1)

        main_layout.addLayout(left_layout, 1)
        main_layout.addWidget(self.video_label, 3)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # 버튼 기능 연결
        self.btn_open_files.clicked.connect(self.open_files)
        self.file_list_widget.itemDoubleClicked.connect(self.play_selected_video)
        self.btn_start_webcam.clicked.connect(self.start_webcam)
        self.btn_stop.clicked.connect(self.stop_video)

        self.thread = None

    def open_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "비디오 파일 선택", "", "Video Files (*.mp4 *.avi)"
        )
        if files:
            self.file_list_widget.clear()
            self.file_list_widget.addItems(files)

    def play_selected_video(self, item):
        self.start_video(item.text())

    def start_webcam(self):
        self.start_video(0)

    def start_video(self, source):
        if self.thread and self.thread.isRunning():
            self.thread.stop()

        # 체크박스 상태를 VideoThread로 전달
        enable_segmentation = self.check_segmentation.isChecked()
        flip_horizontally = self.check_flip.isChecked()
        self.thread = VideoThread(source, enable_segmentation, flip_horizontally)

        self.thread.update_frame.connect(self.update_image)
        self.thread.start()
        self.btn_stop.setEnabled(True)
        self.btn_open_files.setEnabled(False)
        self.btn_start_webcam.setEnabled(False)

    def stop_video(self):
        if self.thread:
            self.thread.stop()
        self.video_label.setText("비디오/웹캠을 시작하세요")
        self.video_label.setStyleSheet(
            "border: 2px solid black; background-color: #333;"
        )
        self.btn_stop.setEnabled(False)
        self.btn_open_files.setEnabled(True)
        self.btn_start_webcam.setEnabled(True)

    def update_image(self, cv_img, status_text):
        if cv_img is not None:
            qt_img = self.convert_cv_qt(cv_img)
            self.video_label.setPixmap(qt_img)
        self.status_bar.showMessage(status_text)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(
            rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        )
        return QPixmap.fromImage(convert_to_Qt_format)

    def closeEvent(self, event):
        self.stop_video()
        event.accept()
