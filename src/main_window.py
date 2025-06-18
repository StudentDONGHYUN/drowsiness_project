# src/main_window.py
import sys
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QListWidget, QFileDialog, QApplication)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
from video_thread import VideoThread

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("운전자 상태 모니터링 시스템")
        self.setGeometry(100, 100, 1200, 800)

        # UI 위젯 생성
        self.video_label = QLabel("비디오/웹캠을 시작하세요", self)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("border: 2px solid black; background-color: #333;")

        self.btn_open_files = QPushButton("비디오 파일 선택", self)
        self.btn_start_webcam = QPushButton("웹캠 시작", self)
        self.btn_stop = QPushButton("중지", self)
        self.btn_stop.setEnabled(False)

        self.file_list_widget = QListWidget(self)
        self.status_bar = self.statusBar()

        # 레이아웃 설정
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()

        left_layout.addWidget(self.file_list_widget)
        left_layout.addWidget(self.btn_open_files)
        left_layout.addWidget(self.btn_start_webcam)
        left_layout.addWidget(self.btn_stop)

        main_layout.addLayout(left_layout, 1) # 왼쪽 영역 비율 1
        main_layout.addWidget(self.video_label, 3) # 오른쪽 비디오 영역 비율 3

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
        files, _ = QFileDialog.getOpenFileNames(self, "비디오 파일 선택", "", "Video Files (*.mp4 *.avi)")
        if files:
            self.file_list_widget.addItems(files)

    def play_selected_video(self, item):
        self.start_video(item.text())

    def start_webcam(self):
        self.start_video(0) # 0은 기본 웹캠 ID

    def start_video(self, source):
        if self.thread and self.thread.isRunning():
            self.thread.stop()

        self.thread = VideoThread(source)
        self.thread.update_frame.connect(self.update_image)
        self.thread.start()
        self.btn_stop.setEnabled(True)
        self.btn_open_files.setEnabled(False)
        self.btn_start_webcam.setEnabled(False)

    def stop_video(self):
        if self.thread:
            self.thread.stop()
        self.video_label.setText("비디오/웹캠을 시작하세요")
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
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def closeEvent(self, event):
        self.stop_video()
        event.accept()
