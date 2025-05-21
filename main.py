import logging
import os.path
import random
import sys
import pickle
import time

import numpy as np
import psola
import soundfile as sf
import sounddevice as sd
import librosa
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QGroupBox, QRadioButton, QComboBox, QDialog
)
from PySide6.QtCore import Qt, QUrl, QTimer, Qt, QThread, Signal, QSize, QCoreApplication

app_name = "AutoTune"
logger = logging.getLogger(app_name)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(
    mode="w",
    filename=app_name + ".log",
    encoding="utf8",
)
fmt = logging.Formatter(
    fmt="%(asctime)s - %(levelname)s - %(message)s"
)
file_handler.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(stream=sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(fmt=fmt)
file_handler.setFormatter(fmt=fmt)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def get_key_frequencies(key) -> list:
    """
    根据调号获取对应的音阶频率

    参数:
    key: 调号，例如:C:maj, C#:maj, F:min 等

    返回:
    包含该调号对应的所有音阶频率的列表
    """
    # 定义基本音阶频率 (A4 = 440Hz)
    base_note_freq = 440
    scale_steps = librosa.key_to_degrees(key)
    # 计算多个八度的音阶频率
    frequencies = []

    for octave in range(1, 6):  # 考虑多个八度
        for step in scale_steps:
            # 计算当前音符的MIDI编号
            midi_number = 12 * (octave + 1) + step
            # 将MIDI编号转换为频率
            frequency = base_note_freq * 2 ** ((midi_number - 69) / 12)
            frequencies.append(frequency)
    frequencies.sort()
    logger.info(f"{key}调的频率：\n{frequencies}")
    return frequencies


class Autotune():

    def __init__(self, input_audio_vocal, input_beat, key, flush=False):
        self.input_audio_vocal = input_audio_vocal
        self.input_beat = input_beat  # 伴奏
        self.key = key
        self.fs_vocal = None  # 采样率,vocal
        self.fs_beat = None
        self.y_vocal = None  # vocal info
        self.y_beat = None
        self.pickle_path_shifted_vocal = "vocal_shifted.pickle"
        self.y_vocal_shifted = None  # 处理之后的人声
        self.frame_length = 1024  # 帧长度
        self.fmin = librosa.note_to_hz("C2")
        self.fmax = librosa.note_to_hz("C7")
        # 定义基本音阶频率 (A4 = 440Hz)
        # self.base_note_freq = 440
        self.flush = flush
        self.autotune_flag = False  # autotune 开关
        self.save_ans = False

    def read_audio(self, sound_type=None):
        """
        :param sound_type: Vocal, beat
        :return:
        """

        pickle_path_vocal = "vocal_info.pickle"
        pickle_path_beat = "beat_info.pickle"

        if sound_type == "Vocal" and sound_type:
            logger.info("读取vocal")
            if self.flush or not os.path.exists(pickle_path_vocal):
                y, fs = sf.read(self.input_audio_vocal)  # y: 音频数据 (numpy array)， fs: 采样率
                logger.info(f"audio length: {len(y)}")
                logger.info(f"vocal原始采样率: {fs} Hz")
                # 2. 将时域信号变为 librosa 格式（浮点数、单声道）
                if y.ndim > 1:
                    y = np.mean(y, axis=1)  # 转为单声道
                y = y.astype(np.float32)
                self.fs_vocal, self.y_vocal = fs, y
                if sound_type is None:
                    logger.info(f" 序列化vocal信息 ")
                    with open(pickle_path_vocal, mode="wb") as f:
                        pickle.dump((self.fs_vocal, self.y_vocal), f)
            else:
                logger.info(f"反序列化vocal信息 ")
                with open(pickle_path_vocal, mode="rb") as f:
                    self.fs_vocal, self.y_vocal = pickle.load(f)

        if sound_type == "Beat" and sound_type:
            logger.info("读取beat")
            if self.flush or not os.path.exists(pickle_path_beat):
                y, fs = sf.read(self.input_beat)  # y: 音频数据 (numpy array)， fs: 采样率
                logger.info(f"audio length: {len(y)}")
                logger.info(f"beat原始采样率: {fs} Hz")
                # 2. 将时域信号变为 librosa 格式（浮点数、单声道）
                if y.ndim > 1:
                    y = np.mean(y, axis=1)  # 转为单声道
                y = y.astype(np.float32)
                self.fs_beat, self.y_beat = fs, y
                if sound_type is None:
                    logger.info(f" 序列化beat信息 ")
                    with open(pickle_path_beat, mode="wb") as f:
                        pickle.dump((self.fs_beat, self.y_beat), f)
            else:
                logger.info(f"反序列化beat信息 ")
                with open(pickle_path_beat, mode="rb") as f:
                    self.fs_beat, self.y_beat = pickle.load(f)
    @staticmethod
    def __play_audio_one(y, fs):
        logger.info("start to play one")
        if y is not None:
            logger.info("start to play one yes")
            sd.play(y, fs)

    def stop_play(self):
        logger.info("stop play")
        sd.stop()

    def play_one_track(self, sound_type):
        """
        :param sound_type: Vocal or Beat
        :return:
        """
        logger.info(f"play one track {sound_type}")
        if sound_type == "Vocal":
            self.__play_audio_one(self.y_vocal_shifted if self.autotune_flag else self.y_vocal, self.fs_vocal)
        else:
            self.__play_audio_one(self.y_beat, self.fs_beat)

    def play_audio_all(self):
        y1 = self.y_vocal_shifted if self.autotune_flag else self.y_vocal
        y2 = self.y_beat
        if y1 is None or y2 is None:
            if y2 is None:
                return self.__play_audio_one(y1, self.fs_vocal)
            else:
                return self.__play_audio_one(y2, self.fs_beat)
        length = max(len(y1), len(y2))  # 对齐长度
        y1 = np.pad(y1, (0, length - len(y1)))
        y2 = np.pad(y2, (0, length - len(y2)))
        # 混合：简单相加并归一化
        mix = y1 + y2
        mix = mix / np.max(np.abs(mix))
        logger.debug(f"混合并播放 Vocal 和 Beat 音轨...")
        self.__play_audio_one(mix, self.fs_vocal)
        # sd.play(mix, self.fs_vocal)
        # sd.wait()

    def autotune_vocal(self, fmin=None, fmax=None):
        logger.info("autotune化vocal")
        if self.y_vocal is None:
            return
        if not self.flush and os.path.exists(self.pickle_path_shifted_vocal):
            logger.info("反序列化shifted vocal")
            with open(self.pickle_path_shifted_vocal, mode="rb") as f:
                self.y_vocal_shifted = pickle.load(f)
            return
        y = self.y_vocal
        if fmin is None and fmax is None:
            fmin, fmax = self.fmin, self.fmax
        # 使用 PYIN 基频估计
        # 使用pYIN算法进行音高检测
        hop_length = self.frame_length // 4  # 帧跳跃长度
        frame_length = self.frame_length
        freqs, voiced_flag, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=self.fs_vocal,
                                             hop_length=hop_length, frame_length=frame_length)
        n_frames = len(freqs)
        logger.info(f"freqs长度:{n_frames}")
        duration = n_frames * hop_length / self.fs_vocal
        logger.info(f"vocal总时长：{duration} s")

        key_frequencies = get_key_frequencies(self.key)
        print(f"获取目标调号的所有音阶频率{key_frequencies}")
        # 对每个时间点的频率进行校正
        corrected_frequency = np.zeros_like(freqs)
        for idx, freq in enumerate(freqs):
            if np.isnan(freq) or freq < 100 or freq > 10000 or voiced_flag[idx] < 0.5:
                corrected_frequency[idx] = freq
                continue
            # 寻找最接近的调内音阶频率
            closest_key_freq = min(key_frequencies, key=lambda x: abs(np.log2(x / freq)))
            # closest_key_freq = key_frequencies[key_frequencies.index(closest_key_freq) + (1 if key_frequencies.index(closest_key_freq)%2 else -1)]
            if abs(closest_key_freq-freq) < 0:
                corrected_frequency[idx] = freq
            else:
                corrected_frequency[idx] = closest_key_freq
                logger.info(f"矫正频率：{freq} hz to {corrected_frequency[idx]} hz")
            # corrected_frequency[idx] = 184
        # 将频率变成波形, 最核心的一行代码🤣
        self.y_vocal_shifted = psola.vocode(self.y_vocal, sample_rate=self.fs_vocal,
                                            target_pitch=corrected_frequency, fmin=fmin, fmax=fmax)
        # 序列化
        if self.save_ans:
            logger.info("序列化shifted vocal")
            with open(self.pickle_path_shifted_vocal, mode="wb") as f:
                pickle.dump(self.y_vocal_shifted, f)

    def main_work(self):
        self.read_audio()
        self.autotune_vocal()
        self.play_audio_all()


class RunThread(QThread):
    finished = Signal(tuple)

    def __init__(self, autotune=None):
        super().__init__()
        if autotune is None:
            self.autotune = Autotune(
                input_beat="",
                input_audio_vocal="",
                key="C:maj",
                flush=True,
            )
        else:
            self.autotune = autotune

    def run(self):
        self.autotune.autotune_vocal()
        self.finished.emit((self.autotune,))


# if __name__ == "__main__":
#     tangyijun_autotune = TangYijunAutotune(
#         input_audio_vocal="看月亮爬上来vocal.mp3",
#         input_beat="看月亮爬上来伴奏.mp3",
#         key="F#:maj",
#         flush=False
#     )
#     tangyijun_autotune.main_work()


class AudioPlayerUI(QWidget):
    def __init__(self):
        super().__init__()
        self.key = "C"
        self.key_type = "maj"
        self.setWindowTitle("唐义军的autotune")
        self.setMinimumSize(400, 200)
        layout = QVBoxLayout()

        # 创建两个音频组
        self.player_widgets = []
        temp = {
            0: "Vocal",
            1: "Beat"
        }
        self.autotune = Autotune(
            input_beat="",
            input_audio_vocal="",
            key="C:maj",
            flush=True,
        )
        self.thread = None
        # self.run_thread = RunThread()
        # 音频key信息
        group_layout0 = QHBoxLayout()
        # 第一个下拉
        self.label1 = QLabel("调：")
        self.combo1 = QComboBox()
        self.combo1.addItems(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",])
        self.combo1.currentTextChanged.connect(self.on_select1)

        # 第二个下拉
        self.label2 = QLabel("调式：")
        self.combo2 = QComboBox()
        self.combo2.addItems(["maj", "min"])
        self.combo2.currentTextChanged.connect(self.on_select2)

        group_layout0.addWidget(self.label1)
        group_layout0.addWidget(self.combo1)
        group_layout0.addWidget(self.label2)
        group_layout0.addWidget(self.combo2)
        layout.addLayout(group_layout0)
        layout.addStretch()
        for i in range(2):
            group = QGroupBox(f"音频通道 {i+1}_{temp[i]}")
            group_layout = QHBoxLayout()

            label = QLabel("未选择音频")
            btn_select = QPushButton("选择音频")
            btn_play = QPushButton("播放")
            btn_remove = QPushButton("Remove")

            # 绑定信号
            btn_select.clicked.connect(lambda _, ii=i, lb=label: self.select_file(sound_type=temp[ii], label=lb))
            btn_play.clicked.connect(lambda _, ii=i: self.autotune.play_one_track(sound_type=temp[ii]))
            btn_remove.clicked.connect(lambda _, ii=i, lb=label: self.remove_sound_file(sound_type=temp[ii], label=lb))

            group_layout.addWidget(label)
            group_layout.addWidget(btn_select)
            group_layout.addWidget(btn_play)
            group_layout.addWidget(btn_remove)
            group.setLayout(group_layout)
            layout.addWidget(group)
        # 添加播放全部按钮
        final_h_layout = QHBoxLayout()
        btn_play_all = QPushButton("播放全部")
        btn_stop_play_all = QPushButton("停止播放")
        autotune_on = QRadioButton("Vocal autotune效果")

        # 绑定信号
        btn_play_all.clicked.connect(lambda _: self.autotune.play_audio_all())
        btn_stop_play_all.clicked.connect(lambda _: self.autotune.stop_play())
        autotune_on.toggled.connect(self.on_toggled)
        final_h_layout.addWidget(btn_play_all)
        final_h_layout.addWidget(btn_stop_play_all)
        final_h_layout.addWidget(autotune_on)
        layout.addLayout(final_h_layout)
        layout.addStretch()
        self.setLayout(layout)

    def on_toggled(self, checked: bool):
        # checked=True 时为“开”，False 时为“关”
        self.autotune.autotune_flag = True if checked else False
        logger.info(f"change autotune flag {self.autotune.autotune_flag}")

    def __update_key(self):
        self.autotune.key = f"{self.key}:{self.key_type}"
        self.__update_autotune()

    def __update_autotune_finish(self, result):
        self.autotune = result[0]
        self.a.close()
        self.a.destroy()
        pass
    def __update_autotune(self):
        self.thread = RunThread(self.autotune)
        self.a = QDialog(self)
        self.thread.finished.connect(self.__update_autotune_finish)
        self.a.setWindowTitle("计算中，请稍等...")
        self.a.setModal(True)
        self.a.setFixedSize(200, 100)
        # self.a.resize(QSize(200, 100))
        # self.setWindow
        self.a.setWindowModality(Qt.ApplicationModal)  # 应用级模态
        # self.a.setWindowFlags(self.a.windowFlags() & ~Qt.WindowType.WindowCloseButtonHint)
        self.thread.start()

        self.a.exec()
        # self.a.show()

    def on_select1(self, text):
        print(f"第一个下拉选择：{text}")
        self.key = text
        self.__update_key()

    def on_select2(self, text):
        print(f"第二个下拉选择：{text}")
        self.key_type = text
        self.__update_key()

    def select_file(self, sound_type, label):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择音频文件", "", "音频文件 (*.mp3 *.wav *.ogg)")
        if file_path:
            url = QUrl.fromLocalFile(file_path)
            logger.info("选择文件：" + url.path() + " type: " + sound_type)
            label.setText(file_path.split('/')[-1])
            if sound_type == "Vocal":
                logger.info("载入Vocal")
                self.autotune.input_audio_vocal = url.path()
                self.autotune.read_audio(sound_type=sound_type)
                # self.autotune.autotune_vocal()
                self.__update_autotune()
            else:
                logger.info("载入beat")
                self.autotune.input_beat = url.path()
                self.autotune.read_audio(sound_type=sound_type)

    def remove_sound_file(self, sound_type, label):
        logger.info(f"remove sound file:{sound_type}")
        if sound_type == "Vocal":
            self.autotune.fs_vocal = None
            self.autotune.y_vocal = None
            self.autotune.input_audio_vocal = None
            self.autotune.y_vocal_shifted = None
        else:
            self.autotune.fs_beat = None
            self.autotune.y_beat = None
            self.autotune.input_beat = None
        label.setText("未选择音频")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioPlayerUI()
    window.show()
    sys.exit(app.exec())
