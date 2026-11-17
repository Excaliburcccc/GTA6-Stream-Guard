"""A reusable native Windows surface for the PROGRAM window."""

import ctypes as ct
from ctypes import wintypes as wt

import cv2
import numpy as np


WNDPROC = ct.WINFUNCTYPE(ct.c_ssize_t, wt.HWND, wt.UINT, ct.c_size_t, ct.c_ssize_t)
WM_PAINT = 0x000F
WM_ERASEBKGND = 0x0014
WS_CHILD = 0x40000000
WS_VISIBLE = 0x10000000
SRCCOPY = 0x00CC0020


class WindowClass(ct.Structure):
    _fields_ = [
        ("style", wt.UINT), ("procedure", WNDPROC),
        ("class_extra", ct.c_int), ("window_extra", ct.c_int),
        ("instance", wt.HINSTANCE), ("icon", wt.HANDLE),
        ("cursor", wt.HANDLE), ("background", wt.HANDLE),
        ("menu", wt.LPCWSTR), ("name", wt.LPCWSTR),
    ]


class Paint(ct.Structure):
    _fields_ = [
        ("dc", wt.HDC), ("erase", wt.BOOL), ("rect", wt.RECT),
        ("restore", wt.BOOL), ("incremental", wt.BOOL),
        ("reserved", ct.c_byte * 32),
    ]


class BitmapInfo(ct.Structure):
    _fields_ = [
        ("size", wt.DWORD), ("width", wt.LONG), ("height", wt.LONG),
        ("planes", wt.WORD), ("bits", wt.WORD), ("compression", wt.DWORD),
        ("image_size", wt.DWORD), ("x_dpi", wt.LONG), ("y_dpi", wt.LONG),
        ("colors_used", wt.DWORD), ("colors_important", wt.DWORD),
    ]


def bind(library, name, result, *arguments):
    function = getattr(library, name)
    function.restype = result
    function.argtypes = arguments
    return function


class WindowsBitmapPresenter:
    """Draw each BGR frame via one reusable top-down bitmap."""

    def __init__(self, root, width, height):
        import tkinter as tk

        self.width = width
        self.height = height
        self._bitmap = self._memory = self._old_bitmap = self._child = None
        self._registered = False
        self._user = ct.WinDLL("user32", use_last_error=True)
        self._gdi = ct.WinDLL("gdi32", use_last_error=True)
        self._kernel = ct.WinDLL("kernel32", use_last_error=True)
        self._bind_native()
        self.widget = tk.Frame(root, width=width, height=height, borderwidth=0,
                               highlightthickness=0, background="black")
        self.widget.pack_propagate(False)
        self.widget.pack()
        try:
            self.widget.update_idletasks()
            self._setup_bitmap()
            self._setup_window(self.widget.winfo_id())
        except Exception:
            self.close()
            self.widget.destroy()
            raise
        self.widget.bind("<Destroy>", self._destroyed, add="+")

    def _bind_native(self):
        user, gdi, kernel = self._user, self._gdi, self._kernel
        hwnd, hdc, handle = wt.HWND, wt.HDC, wt.HANDLE
        i, b, u = ct.c_int, wt.BOOL, wt.UINT
        bind(kernel, "GetModuleHandleW", wt.HMODULE, wt.LPCWSTR)
        bind(user, "RegisterClassW", wt.ATOM, ct.POINTER(WindowClass))
        bind(user, "UnregisterClassW", b, wt.LPCWSTR, wt.HINSTANCE)
        bind(user, "CreateWindowExW", hwnd, wt.DWORD, wt.LPCWSTR, wt.LPCWSTR, wt.DWORD,
             i, i, i, i, hwnd, wt.HMENU, wt.HINSTANCE, ct.c_void_p)
        bind(user, "DefWindowProcW", ct.c_ssize_t, hwnd, u, ct.c_size_t, ct.c_ssize_t)
        bind(user, "DestroyWindow", b, hwnd)
        bind(user, "IsWindow", b, hwnd)
        bind(user, "SetWindowPos", b, hwnd, hwnd, i, i, i, i, u)
        bind(user, "GetDC", hdc, hwnd)
        bind(user, "ReleaseDC", i, hwnd, hdc)
        bind(user, "BeginPaint", hdc, hwnd, ct.POINTER(Paint))
        bind(user, "EndPaint", b, hwnd, ct.POINTER(Paint))
        bind(gdi, "CreateCompatibleDC", hdc, hdc)
        bind(gdi, "DeleteDC", b, hdc)
        bind(gdi, "CreateDIBSection", handle, hdc, ct.POINTER(BitmapInfo),
             u, ct.POINTER(ct.c_void_p), handle, wt.DWORD)
        bind(gdi, "SelectObject", handle, hdc, handle)
        bind(gdi, "DeleteObject", b, handle)
        bind(gdi, "BitBlt", b, hdc, i, i, i, i, hdc, i, i, wt.DWORD)
        bind(gdi, "GdiFlush", b)

    def _setup_bitmap(self):
        self._memory = self._gdi.CreateCompatibleDC(None)
        if not self._memory:
            raise ct.WinError(ct.get_last_error())
        info = BitmapInfo()
        info.size = ct.sizeof(info)
        info.width, info.height = self.width, -self.height
        info.planes, info.bits = 1, 32
        address = ct.c_void_p()
        self._bitmap = self._gdi.CreateDIBSection(self._memory, ct.byref(info), 0,
                                                   ct.byref(address), None, 0)
        if not self._bitmap or not address.value:
            raise ct.WinError(ct.get_last_error())
        self._old_bitmap = self._gdi.SelectObject(self._memory, self._bitmap)
        if not self._old_bitmap or self._old_bitmap == ct.c_void_p(-1).value:
            self._old_bitmap = None
            raise ct.WinError(ct.get_last_error())
        data = (ct.c_uint8 * (self.height * self.width * 4)).from_address(address.value)
        self._pixels = np.ctypeslib.as_array(data).reshape(self.height, self.width, 4)
        self._pixels.fill(0)

    def _setup_window(self, parent):
        self._instance = self._kernel.GetModuleHandleW(None)
        self._name = f"GTA6GuardDIB_{id(self):x}"
        self._window_proc = WNDPROC(self._process_message)
        definition = WindowClass()
        definition.procedure = self._window_proc
        definition.instance = self._instance
        definition.name = self._name
        if not self._user.RegisterClassW(ct.byref(definition)):
            raise ct.WinError(ct.get_last_error())
        self._registered = True
        self._child = self._user.CreateWindowExW(
            0, self._name, "", WS_CHILD | WS_VISIBLE,
            0, 0, self.width, self.height, parent, None, self._instance, None,
        )
        if not self._child:
            raise ct.WinError(ct.get_last_error())
        # Keep the video filling the Tk placeholder when its mapped geometry settles.
        self._user.SetWindowPos(self._child, None, 0, 0, self.width, self.height, 0x14)

    def _blit(self, target):
        if target and self._memory:
            self._gdi.BitBlt(target, 0, 0, self.width, self.height,
                            self._memory, 0, 0, SRCCOPY)
            # Synchronize the GDI reads before rewriting the backing pixels.
            self._gdi.GdiFlush()

    def _process_message(self, hwnd, message, wparam, lparam):
        if message == WM_ERASEBKGND:
            return 1
        if message == WM_PAINT:
            paint = Paint()
            dc = self._user.BeginPaint(hwnd, ct.byref(paint))
            try:
                self._blit(dc)
            finally:
                self._user.EndPaint(hwnd, ct.byref(paint))
            return 0
        return self._user.DefWindowProcW(hwnd, message, wparam, lparam)

    def show(self, frame):
        if frame.shape[:2] != (self.height, self.width):
            from app import fit
            frame = fit(frame, (self.width, self.height))
        cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA, dst=self._pixels)
        dc = self._user.GetDC(self._child)
        try:
            self._blit(dc)
        finally:
            if dc:
                self._user.ReleaseDC(self._child, dc)

    def _destroyed(self, event):
        if event.widget is self.widget:
            self.close()

    def close(self):
        if self._child and self._user.IsWindow(self._child):
            self._user.DestroyWindow(self._child)
        self._child = None
        if self._registered:
            self._user.UnregisterClassW(self._name, self._instance)
            self._registered = False
        if self._old_bitmap:
            self._gdi.SelectObject(self._memory, self._old_bitmap)
            self._old_bitmap = None
        if self._bitmap:
            self._gdi.DeleteObject(self._bitmap)
            self._bitmap = None
        if self._memory:
            self._gdi.DeleteDC(self._memory)
            self._memory = None
