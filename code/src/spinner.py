"""
Module: spinner.py
Chức năng: Cung cấp hoạt ảnh spinner chạy trên terminal trong khi
           các tác vụ nặng (GridSearchCV, fit model, visualization...) đang thực hiện.

Sử dụng:
    with Spinner("Dang huan luyen mo hinh"):
        heavy_function()
"""

import sys
import time
import threading
import itertools


class Spinner:
    """
    Context Manager hiển thị spinner xoay vòng + thông báo trạng thái
    trong khi một tác vụ đang chạy.

    Ví dụ:
        with Spinner("Dang chay GridSearchCV"):
            grid_search.fit(X_train, y_train)
    """

    # Các kiểu hoạt ảnh có sẵn
    STYLES = {
        'dots'   : ['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷'],
        'classic': ['|', '/', '-', '\\'],
        'arrow'  : ['←', '↖', '↑', '↗', '→', '↘', '↓', '↙'],
        'bar'    : ['[=   ]', '[==  ]', '[=== ]', '[====]', '[ ===]', '[  ==]', '[   =]', '[    ]'],
        'bounce' : ['[●    ]', '[ ●   ]', '[  ●  ]', '[   ● ]', '[    ●]', '[   ● ]', '[  ●  ]', '[ ●   ]'],
    }

    def __init__(self, message: str = "Dang xu ly", style: str = 'bounce',
                 delay: float = 0.12, color: bool = True):
        """
        Khởi tạo Spinner.

        Tham số:
            message : Thông báo hiển thị bên cạnh spinner.
            style   : Kiểu hoạt ảnh ('dots', 'classic', 'arrow', 'bar', 'bounce').
            delay   : Tốc độ cập nhật frame (giây).
            color   : Bật màu ANSI trên terminal hỗ trợ màu.
        """
        self.message  = message
        self.frames   = self.STYLES.get(style, self.STYLES['bounce'])
        self.delay    = delay
        self.color    = color
        self._stop    = threading.Event()
        self._thread  = threading.Thread(target=self._spin, daemon=True)
        self._start_t = None

    # --- ANSI color helpers ---
    def _cyan(self, s):   return f"\033[96m{s}\033[0m" if self.color else s
    def _green(self, s):  return f"\033[92m{s}\033[0m" if self.color else s
    def _yellow(self, s): return f"\033[93m{s}\033[0m" if self.color else s
    def _bold(self, s):   return f"\033[1m{s}\033[0m"  if self.color else s

    def _spin(self):
        """Vòng lặp nội bộ: liên tục in frame tiếp theo lên cùng 1 dòng."""
        for frame in itertools.cycle(self.frames):
            if self._stop.is_set():
                break
            elapsed = time.time() - self._start_t
            line = f"\r  {self._cyan(frame)}  {self._bold(self.message)}  {self._yellow(f'({elapsed:.0f}s)')}"
            sys.stdout.write(line)
            sys.stdout.flush()
            time.sleep(self.delay)

    def start(self):
        self._start_t = time.time()
        self._stop.clear()
        self._thread.start()
        return self

    def stop(self, success: bool = True):
        elapsed = time.time() - self._start_t
        self._stop.set()
        self._thread.join()
        # Xoá dòng spinner và in kết quả cuối
        icon = self._green("✔") if success else "\033[91m✘\033[0m"
        sys.stdout.write(
            f"\r  {icon}  {self._bold(self.message)}  "
            f"{self._green(f'(xong trong {elapsed:.1f}s)')}\n"
        )
        sys.stdout.flush()

    # --- Context Manager interface ---
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop(success=(exc_type is None))
        return False   # Không nuốt exception


def spinner_step(message: str, style: str = 'bounce', **kwargs) -> Spinner:
    """
    Shorthand tạo và start Spinner ngay lập tức.
    Dùng khi muốn kiểm soát thủ công (không dùng `with`).

    Ví dụ:
        sp = spinner_step("Dang ve bieu do")
        draw_charts()
        sp.stop()
    """
    s = Spinner(message, style=style, **kwargs)
    s.start()
    return s
