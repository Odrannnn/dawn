// Copyright 2026 The Dawn & Tint Authors
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "src/dawn/native/opengl/EGLOnWGL.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace dawn::native::opengl {

namespace {

// WGL_ARB_create_context / _profile / _robustness and WGL_EXT_swap_control.
constexpr int kWGL_CONTEXT_MAJOR_VERSION_ARB = 0x2091;
constexpr int kWGL_CONTEXT_MINOR_VERSION_ARB = 0x2092;
constexpr int kWGL_CONTEXT_FLAGS_ARB = 0x2094;
constexpr int kWGL_CONTEXT_PROFILE_MASK_ARB = 0x9126;
constexpr int kWGL_CONTEXT_CORE_PROFILE_BIT_ARB = 0x1;
constexpr int kWGL_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB = 0x2;
constexpr int kWGL_CONTEXT_DEBUG_BIT_ARB = 0x1;
constexpr int kWGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB = 0x2;
constexpr int kWGL_CONTEXT_ROBUST_ACCESS_BIT_ARB = 0x4;
constexpr int kWGL_CONTEXT_RESET_NOTIFICATION_STRATEGY_ARB = 0x8256;
constexpr int kWGL_LOSE_CONTEXT_ON_RESET_ARB = 0x8252;
constexpr int kWGL_NO_RESET_NOTIFICATION_ARB = 0x8261;

// GL sync objects (core since 3.2), used to implement EGL_KHR_fence_sync.
constexpr unsigned kGL_SYNC_GPU_COMMANDS_COMPLETE = 0x9117;
constexpr unsigned kGL_SYNC_FLUSH_COMMANDS_BIT = 0x1;
constexpr unsigned kGL_SYNC_STATUS = 0x9114;
constexpr int kGL_SIGNALED = 0x9119;
constexpr unsigned kGL_ALREADY_SIGNALED = 0x911A;
constexpr unsigned kGL_TIMEOUT_EXPIRED = 0x911B;
constexpr unsigned kGL_CONDITION_SATISFIED = 0x911C;

using GLsyncHandle = void*;
using PFN_glFenceSync = GLsyncHandle(APIENTRY*)(unsigned condition, unsigned flags);
using PFN_glClientWaitSync = unsigned(APIENTRY*)(GLsyncHandle sync,
                                                 unsigned flags,
                                                 uint64_t timeout);
using PFN_glDeleteSync = void(APIENTRY*)(GLsyncHandle sync);
using PFN_glGetSynciv = void(APIENTRY*)(GLsyncHandle sync,
                                        unsigned pname,
                                        int bufSize,
                                        int* length,
                                        int* values);

// opengl32.dll is loaded at run time so that dawn_native doesn't link against it.
using PFN_wglCreateContext = HGLRC(WINAPI*)(HDC);
using PFN_wglDeleteContext = BOOL(WINAPI*)(HGLRC);
using PFN_wglMakeCurrent = BOOL(WINAPI*)(HDC, HGLRC);
using PFN_wglGetProcAddress = PROC(WINAPI*)(LPCSTR);
using PFN_wglGetCurrentContext = HGLRC(WINAPI*)();
using PFN_wglGetCurrentDC = HDC(WINAPI*)();
using PFN_wglCreateContextAttribsARB = HGLRC(WINAPI*)(HDC, HGLRC, const int*);
using PFN_wglGetExtensionsStringARB = const char*(WINAPI*)(HDC);
using PFN_wglSwapIntervalEXT = BOOL(WINAPI*)(int);

constexpr wchar_t kWindowClass[] = L"DawnEGLOnWGL";
constexpr char kDisplayExtensions[] =
    "EGL_KHR_create_context EGL_EXT_create_context_robustness EGL_KHR_fence_sync "
    "EGL_KHR_surfaceless_context";

EGLDisplay const kDisplay = reinterpret_cast<EGLDisplay>(uintptr_t(1));
EGLConfig const kConfig = reinterpret_cast<EGLConfig>(uintptr_t(1));

struct Surface {
    HWND hwnd = nullptr;
    HDC dc = nullptr;
    bool ownsWindow = false;
    int appliedSwapInterval = -1;
};

struct Context {
    HGLRC rc = nullptr;
};

struct ThreadState {
    Context* context = nullptr;
    Surface* draw = nullptr;
    Surface* read = nullptr;
    EGLint error = EGL_SUCCESS;
};

thread_local ThreadState tState;

struct Display {
    std::mutex mutex;
    bool initialized = false;

    HMODULE opengl32 = nullptr;
    PFN_wglCreateContext createContext = nullptr;
    PFN_wglDeleteContext deleteContext = nullptr;
    PFN_wglMakeCurrent makeCurrent = nullptr;
    PFN_wglGetProcAddress getProcAddress = nullptr;
    PFN_wglGetCurrentContext getCurrentContext = nullptr;
    PFN_wglGetCurrentDC getCurrentDC = nullptr;
    PFN_wglCreateContextAttribsARB createContextAttribs = nullptr;
    PFN_wglSwapIntervalEXT swapInterval = nullptr;
    bool hasRobustness = false;

    // A hidden window whose DC is used for surfaceless contexts on the thread that initialized
    // the display; other threads get their own (SurfacelessDC).
    HWND hiddenWindow = nullptr;
    HDC hiddenDC = nullptr;
    DWORD hiddenThread = 0;

    // eglSwapInterval applies to the next swap of each surface; Dawn sets it outside of a
    // current scope, and wglSwapIntervalEXT needs one.
    std::atomic<int> desiredSwapInterval{1};

    std::once_flag glProcsOnce;
    PFN_glFenceSync fenceSync = nullptr;
    PFN_glClientWaitSync clientWaitSync = nullptr;
    PFN_glDeleteSync deleteSync = nullptr;
    PFN_glGetSynciv getSynciv = nullptr;

    // Syncs destroyed while no context was current; deleted at the next MakeCurrent.
    std::vector<GLsyncHandle> pendingDeletes;
};

Display& GetDisplayState() {
    static Display* display = new Display();
    return *display;
}

EGLBoolean SetError(EGLint error) {
    tState.error = error;
    return error == EGL_SUCCESS ? EGL_TRUE : EGL_FALSE;
}

bool IsValidDisplay(EGLDisplay dpy) {
    if (dpy != kDisplay) {
        SetError(EGL_BAD_DISPLAY);
        return false;
    }
    if (!GetDisplayState().initialized) {
        SetError(EGL_NOT_INITIALIZED);
        return false;
    }
    return true;
}

PROC LookupGLProc(const char* name) {
    Display& d = GetDisplayState();
    if (d.getProcAddress != nullptr) {
        PROC proc = d.getProcAddress(name);
        // wglGetProcAddress may return these small values instead of nullptr on failure.
        intptr_t value = reinterpret_cast<intptr_t>(proc);
        if (value != 0 && value != 1 && value != 2 && value != 3 && value != -1) {
            return proc;
        }
    }
    // GL 1.1 entry points are only exported by opengl32.dll.
    return d.opengl32 != nullptr ? ::GetProcAddress(d.opengl32, name) : nullptr;
}

void LoadGLProcsOnce() {
    Display& d = GetDisplayState();
    std::call_once(d.glProcsOnce, [&] {
        d.fenceSync = reinterpret_cast<PFN_glFenceSync>(LookupGLProc("glFenceSync"));
        d.clientWaitSync = reinterpret_cast<PFN_glClientWaitSync>(LookupGLProc("glClientWaitSync"));
        d.deleteSync = reinterpret_cast<PFN_glDeleteSync>(LookupGLProc("glDeleteSync"));
        d.getSynciv = reinterpret_cast<PFN_glGetSynciv>(LookupGLProc("glGetSynciv"));
    });
}

bool SetPixelFormatOnce(HDC dc) {
    // Keep whatever format a window already has: SetPixelFormat can only be called once per
    // window, and a mismatch makes wglMakeCurrent fail, which is reported there.
    if (::GetPixelFormat(dc) != 0) {
        return true;
    }
    PIXELFORMATDESCRIPTOR pfd = {};
    pfd.nSize = sizeof(pfd);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    pfd.cAlphaBits = 8;
    pfd.iLayerType = PFD_MAIN_PLANE;
    int format = ::ChoosePixelFormat(dc, &pfd);
    return format != 0 && ::SetPixelFormat(dc, format, &pfd);
}

HWND CreateHiddenWindow() {
    static std::once_flag registered;
    HINSTANCE instance = ::GetModuleHandleW(nullptr);
    std::call_once(registered, [&] {
        WNDCLASSEXW wc = {};
        wc.cbSize = sizeof(wc);
        wc.style = CS_OWNDC;
        wc.lpfnWndProc = ::DefWindowProcW;
        wc.hInstance = instance;
        wc.lpszClassName = kWindowClass;
        ::RegisterClassExW(&wc);
    });
    return ::CreateWindowExW(0, kWindowClass, L"", WS_OVERLAPPEDWINDOW, 0, 0, 1, 1, nullptr,
                             nullptr, instance, nullptr);
}

bool HasWGLExtension(const char* extensions, const char* name) {
    if (extensions == nullptr) {
        return false;
    }
    size_t length = std::strlen(name);
    for (const char* p = extensions; (p = std::strstr(p, name)) != nullptr; p += length) {
        bool startOk = p == extensions || p[-1] == ' ';
        bool endOk = p[length] == '\0' || p[length] == ' ';
        if (startOk && endOk) {
            return true;
        }
    }
    return false;
}

// Loads opengl32.dll and the WGL extension entry points, which need a current legacy context.
bool InitializeDisplay(Display& d) {
    d.opengl32 = ::LoadLibraryW(L"opengl32.dll");
    if (d.opengl32 == nullptr) {
        return false;
    }
    auto get = [&](const char* name) { return ::GetProcAddress(d.opengl32, name); };
    d.createContext = reinterpret_cast<PFN_wglCreateContext>(get("wglCreateContext"));
    d.deleteContext = reinterpret_cast<PFN_wglDeleteContext>(get("wglDeleteContext"));
    d.makeCurrent = reinterpret_cast<PFN_wglMakeCurrent>(get("wglMakeCurrent"));
    d.getProcAddress = reinterpret_cast<PFN_wglGetProcAddress>(get("wglGetProcAddress"));
    d.getCurrentContext = reinterpret_cast<PFN_wglGetCurrentContext>(get("wglGetCurrentContext"));
    d.getCurrentDC = reinterpret_cast<PFN_wglGetCurrentDC>(get("wglGetCurrentDC"));
    if (d.createContext == nullptr || d.deleteContext == nullptr || d.makeCurrent == nullptr ||
        d.getProcAddress == nullptr || d.getCurrentContext == nullptr ||
        d.getCurrentDC == nullptr) {
        return false;
    }

    d.hiddenWindow = CreateHiddenWindow();
    if (d.hiddenWindow == nullptr) {
        return false;
    }
    d.hiddenDC = ::GetDC(d.hiddenWindow);
    if (d.hiddenDC == nullptr || !SetPixelFormatOnce(d.hiddenDC)) {
        return false;
    }
    d.hiddenThread = ::GetCurrentThreadId();

    HGLRC bootstrap = d.createContext(d.hiddenDC);
    if (bootstrap == nullptr) {
        return false;
    }
    HGLRC previousContext = d.getCurrentContext();
    HDC previousDC = d.getCurrentDC();
    if (!d.makeCurrent(d.hiddenDC, bootstrap)) {
        d.deleteContext(bootstrap);
        return false;
    }

    d.createContextAttribs =
        reinterpret_cast<PFN_wglCreateContextAttribsARB>(LookupGLProc("wglCreateContextAttribsARB"));
    d.swapInterval = reinterpret_cast<PFN_wglSwapIntervalEXT>(LookupGLProc("wglSwapIntervalEXT"));
    auto getExtensions =
        reinterpret_cast<PFN_wglGetExtensionsStringARB>(LookupGLProc("wglGetExtensionsStringARB"));
    const char* extensions = getExtensions != nullptr ? getExtensions(d.hiddenDC) : nullptr;
    d.hasRobustness = HasWGLExtension(extensions, "WGL_ARB_create_context_robustness");

    d.makeCurrent(previousDC, previousContext);
    d.deleteContext(bootstrap);

    // Core contexts of a given version can only be made with WGL_ARB_create_context.
    return d.createContextAttribs != nullptr;
}

// A hidden window owned by one thread, destroyed when the thread exits.
struct ThreadWindow {
    HWND hwnd = nullptr;
    HDC dc = nullptr;
    ~ThreadWindow() {
        if (hwnd == nullptr) {
            return;
        }
        Display& d = GetDisplayState();
        if (d.getCurrentDC != nullptr && d.getCurrentDC() == dc) {
            d.makeCurrent(nullptr, nullptr);
        }
        ::ReleaseDC(hwnd, dc);
        ::DestroyWindow(hwnd);
    }
};

thread_local ThreadWindow tWindow;

// The DC a surfaceless context is made current on. WGL needs a DC, and two threads with
// contexts current on one DC at once make some drivers (AMD) fail wglMakeCurrent, so each
// thread gets its own hidden window, with the pixel format of the display's.
HDC SurfacelessDC(Display& d) {
    if (::GetCurrentThreadId() == d.hiddenThread) {
        return d.hiddenDC;
    }
    if (tWindow.dc == nullptr) {
        HWND hwnd = CreateHiddenWindow();
        HDC dc = hwnd != nullptr ? ::GetDC(hwnd) : nullptr;
        int format = ::GetPixelFormat(d.hiddenDC);
        PIXELFORMATDESCRIPTOR pfd = {};
        if (dc == nullptr || format == 0 ||
            ::DescribePixelFormat(d.hiddenDC, format, sizeof(pfd), &pfd) == 0 ||
            !::SetPixelFormat(dc, format, &pfd)) {
            if (dc != nullptr) {
                ::ReleaseDC(hwnd, dc);
            }
            if (hwnd != nullptr) {
                ::DestroyWindow(hwnd);
            }
            return d.hiddenDC;
        }
        tWindow.hwnd = hwnd;
        tWindow.dc = dc;
    }
    return tWindow.dc;
}

// wglMakeCurrent, returning its Win32 error (0 on success) and logging a failure, since an EGL
// error code alone can't tell a context busy on another thread from a pixel format mismatch.
DWORD MakeCurrentLogged(Display& d, HDC dc, HGLRC rc) {
    if (d.makeCurrent(dc, rc)) {
        return 0;
    }
    DWORD error = ::GetLastError();
    std::fprintf(stderr,
                 "EGL-on-WGL: wglMakeCurrent(dc %p, rc %p) failed on thread %lu: Win32 error "
                 "%lu (0x%lx)\n",
                 static_cast<void*>(dc), static_cast<void*>(rc), ::GetCurrentThreadId(), error,
                 error);
    std::fflush(stderr);
    return error != 0 ? error : ERROR_INVALID_FUNCTION;
}

void FlushPendingDeletes(Display& d) {
    std::vector<GLsyncHandle> pending;
    {
        std::lock_guard<std::mutex> lock(d.mutex);
        pending.swap(d.pendingDeletes);
    }
    if (d.deleteSync != nullptr) {
        for (GLsyncHandle sync : pending) {
            d.deleteSync(sync);
        }
    }
}

// EGL 1.0

EGLBoolean EGLAPIENTRY ChooseConfig(EGLDisplay dpy,
                                    const EGLint* attribs,
                                    EGLConfig* configs,
                                    EGLint configSize,
                                    EGLint* numConfig) {
    if (!IsValidDisplay(dpy)) {
        return EGL_FALSE;
    }
    if (numConfig == nullptr) {
        return SetError(EGL_BAD_PARAMETER);
    }
    bool matches = true;
    for (const EGLint* a = attribs; a != nullptr && a[0] != EGL_NONE; a += 2) {
        EGLint value = a[1];
        if (value == EGL_DONT_CARE) {
            continue;
        }
        switch (a[0]) {
            case EGL_RED_SIZE:
            case EGL_GREEN_SIZE:
            case EGL_BLUE_SIZE:
            case EGL_ALPHA_SIZE:
                matches &= value <= 8;
                break;
            case EGL_BUFFER_SIZE:
                matches &= value <= 32;
                break;
            case EGL_DEPTH_SIZE:
            case EGL_STENCIL_SIZE:
            case EGL_SAMPLES:
            case EGL_SAMPLE_BUFFERS:
                matches &= value == 0;
                break;
            case EGL_SURFACE_TYPE:
                matches &= (value & ~(EGL_WINDOW_BIT | EGL_PBUFFER_BIT)) == 0;
                break;
            case EGL_RENDERABLE_TYPE:
            case EGL_CONFORMANT:
                matches &= (value & ~EGL_OPENGL_BIT) == 0;
                break;
            case EGL_COLOR_BUFFER_TYPE:
                matches &= value == EGL_RGB_BUFFER;
                break;
            case EGL_COLOR_COMPONENT_TYPE_EXT:
                matches &= value == EGL_COLOR_COMPONENT_TYPE_FIXED_EXT;
                break;
            case EGL_CONFIG_ID:
                matches &= value == 1;
                break;
            default:
                break;
        }
    }
    *numConfig = matches ? 1 : 0;
    if (matches && configs != nullptr && configSize >= 1) {
        configs[0] = kConfig;
    }
    return SetError(EGL_SUCCESS);
}

EGLBoolean EGLAPIENTRY CopyBuffers(EGLDisplay, EGLSurface, EGLNativePixmapType) {
    return SetError(EGL_BAD_NATIVE_PIXMAP);
}

EGLContext EGLAPIENTRY CreateContext(EGLDisplay dpy,
                                     EGLConfig config,
                                     EGLContext share,
                                     const EGLint* attribs) {
    if (!IsValidDisplay(dpy)) {
        return EGL_NO_CONTEXT;
    }
    if (config != kConfig && config != EGL_NO_CONFIG_KHR) {
        SetError(EGL_BAD_CONFIG);
        return EGL_NO_CONTEXT;
    }
    Display& d = GetDisplayState();

    int major = 1;
    int minor = 0;
    int flags = 0;
    int profile = kWGL_CONTEXT_CORE_PROFILE_BIT_ARB;
    int resetStrategy = kWGL_NO_RESET_NOTIFICATION_ARB;
    for (const EGLint* a = attribs; a != nullptr && a[0] != EGL_NONE; a += 2) {
        EGLint value = a[1];
        switch (a[0]) {
            case EGL_CONTEXT_MAJOR_VERSION_KHR:
                major = value;
                break;
            case EGL_CONTEXT_MINOR_VERSION_KHR:
                minor = value;
                break;
            case EGL_CONTEXT_FLAGS_KHR:
                if (value & EGL_CONTEXT_OPENGL_DEBUG_BIT_KHR) {
                    flags |= kWGL_CONTEXT_DEBUG_BIT_ARB;
                }
                if (value & EGL_CONTEXT_OPENGL_FORWARD_COMPATIBLE_BIT_KHR) {
                    flags |= kWGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB;
                }
                if (value & EGL_CONTEXT_OPENGL_ROBUST_ACCESS_BIT_KHR) {
                    flags |= kWGL_CONTEXT_ROBUST_ACCESS_BIT_ARB;
                }
                break;
            case EGL_CONTEXT_OPENGL_PROFILE_MASK_KHR:
                profile = (value & EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT_KHR)
                              ? kWGL_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB
                              : kWGL_CONTEXT_CORE_PROFILE_BIT_ARB;
                break;
            case EGL_CONTEXT_OPENGL_ROBUST_ACCESS_EXT:
                if (value == EGL_TRUE) {
                    flags |= kWGL_CONTEXT_ROBUST_ACCESS_BIT_ARB;
                }
                break;
            case EGL_CONTEXT_OPENGL_RESET_NOTIFICATION_STRATEGY_EXT:
                resetStrategy = value == EGL_LOSE_CONTEXT_ON_RESET_EXT
                                    ? kWGL_LOSE_CONTEXT_ON_RESET_ARB
                                    : kWGL_NO_RESET_NOTIFICATION_ARB;
                break;
            default:
                SetError(EGL_BAD_ATTRIBUTE);
                return EGL_NO_CONTEXT;
        }
    }
    // Robust access is advertised unconditionally because Dawn requires it, but drivers without
    // WGL_ARB_create_context_robustness reject the flag: drop it there rather than fail.
    if (!d.hasRobustness) {
        flags &= ~kWGL_CONTEXT_ROBUST_ACCESS_BIT_ARB;
    }

    std::vector<int> wglAttribs = {
        kWGL_CONTEXT_MAJOR_VERSION_ARB, major, kWGL_CONTEXT_MINOR_VERSION_ARB, minor,
        kWGL_CONTEXT_FLAGS_ARB,         flags, kWGL_CONTEXT_PROFILE_MASK_ARB,  profile,
    };
    if (d.hasRobustness) {
        wglAttribs.push_back(kWGL_CONTEXT_RESET_NOTIFICATION_STRATEGY_ARB);
        wglAttribs.push_back(resetStrategy);
    }
    wglAttribs.push_back(0);

    HGLRC shareRC = share != EGL_NO_CONTEXT ? static_cast<Context*>(share)->rc : nullptr;
    // The requested major/minor go straight to WGL. A failure (including a driver that refuses the
    // version) must be reported as EGL_NO_CONTEXT rather than falling back to an older context
    // here: eglCreateContext callers retry with a lower version themselves (see ContextEGL).
    HGLRC rc = d.createContextAttribs(SurfacelessDC(d), shareRC, wglAttribs.data());
    if (rc == nullptr) {
        SetError(EGL_BAD_MATCH);
        return EGL_NO_CONTEXT;
    }
    Context* context = new Context();
    context->rc = rc;
    SetError(EGL_SUCCESS);
    return context;
}

EGLSurface EGLAPIENTRY CreatePbufferSurface(EGLDisplay dpy, EGLConfig config, const EGLint*) {
    if (!IsValidDisplay(dpy)) {
        return EGL_NO_SURFACE;
    }
    if (config != kConfig) {
        SetError(EGL_BAD_CONFIG);
        return EGL_NO_SURFACE;
    }
    // Dawn only uses a pbuffer as a stand-in for surfaceless contexts; a hidden window does.
    HWND hwnd = CreateHiddenWindow();
    HDC dc = hwnd != nullptr ? ::GetDC(hwnd) : nullptr;
    if (dc == nullptr || !SetPixelFormatOnce(dc)) {
        if (hwnd != nullptr) {
            ::DestroyWindow(hwnd);
        }
        SetError(EGL_BAD_ALLOC);
        return EGL_NO_SURFACE;
    }
    Surface* surface = new Surface();
    surface->hwnd = hwnd;
    surface->dc = dc;
    surface->ownsWindow = true;
    SetError(EGL_SUCCESS);
    return surface;
}

EGLSurface EGLAPIENTRY CreatePixmapSurface(EGLDisplay, EGLConfig, EGLNativePixmapType, const EGLint*) {
    SetError(EGL_BAD_MATCH);
    return EGL_NO_SURFACE;
}

EGLSurface EGLAPIENTRY CreateWindowSurface(EGLDisplay dpy,
                                           EGLConfig config,
                                           EGLNativeWindowType window,
                                           const EGLint* attribs) {
    if (!IsValidDisplay(dpy)) {
        return EGL_NO_SURFACE;
    }
    if (config != kConfig) {
        SetError(EGL_BAD_CONFIG);
        return EGL_NO_SURFACE;
    }
    for (const EGLint* a = attribs; a != nullptr && a[0] != EGL_NONE; a += 2) {
        // Only the default linear colorspace and back buffer rendering are supported.
        if (a[0] == EGL_GL_COLORSPACE_KHR && a[1] != EGL_GL_COLORSPACE_LINEAR_KHR) {
            SetError(EGL_BAD_MATCH);
            return EGL_NO_SURFACE;
        }
    }
    HWND hwnd = static_cast<HWND>(window);
    if (hwnd == nullptr || !::IsWindow(hwnd)) {
        SetError(EGL_BAD_NATIVE_WINDOW);
        return EGL_NO_SURFACE;
    }
    HDC dc = ::GetDC(hwnd);
    if (dc == nullptr || !SetPixelFormatOnce(dc)) {
        if (dc != nullptr) {
            ::ReleaseDC(hwnd, dc);
        }
        SetError(EGL_BAD_NATIVE_WINDOW);
        return EGL_NO_SURFACE;
    }
    Surface* surface = new Surface();
    surface->hwnd = hwnd;
    surface->dc = dc;
    SetError(EGL_SUCCESS);
    return surface;
}

EGLBoolean EGLAPIENTRY DestroyContext(EGLDisplay dpy, EGLContext ctx) {
    if (!IsValidDisplay(dpy)) {
        return EGL_FALSE;
    }
    if (ctx == EGL_NO_CONTEXT) {
        return SetError(EGL_BAD_CONTEXT);
    }
    Context* context = static_cast<Context*>(ctx);
    if (tState.context == context) {
        GetDisplayState().makeCurrent(nullptr, nullptr);
        tState.context = nullptr;
        tState.draw = nullptr;
        tState.read = nullptr;
    }
    GetDisplayState().deleteContext(context->rc);
    delete context;
    return SetError(EGL_SUCCESS);
}

EGLBoolean EGLAPIENTRY DestroySurface(EGLDisplay dpy, EGLSurface surf) {
    if (!IsValidDisplay(dpy)) {
        return EGL_FALSE;
    }
    if (surf == EGL_NO_SURFACE) {
        return SetError(EGL_BAD_SURFACE);
    }
    Surface* surface = static_cast<Surface*>(surf);
    if (tState.draw == surface || tState.read == surface) {
        // EGL defers destruction of a current surface; rebind the context surfaceless instead.
        Display& d = GetDisplayState();
        d.makeCurrent(tState.context != nullptr ? SurfacelessDC(d) : nullptr,
                      tState.context != nullptr ? tState.context->rc : nullptr);
        tState.draw = nullptr;
        tState.read = nullptr;
    }
    ::ReleaseDC(surface->hwnd, surface->dc);
    if (surface->ownsWindow) {
        ::DestroyWindow(surface->hwnd);
    }
    delete surface;
    return SetError(EGL_SUCCESS);
}

EGLBoolean EGLAPIENTRY GetConfigAttrib(EGLDisplay dpy,
                                       EGLConfig config,
                                       EGLint attribute,
                                       EGLint* value) {
    if (!IsValidDisplay(dpy)) {
        return EGL_FALSE;
    }
    if (config != kConfig) {
        return SetError(EGL_BAD_CONFIG);
    }
    if (value == nullptr) {
        return SetError(EGL_BAD_PARAMETER);
    }
    switch (attribute) {
        case EGL_BUFFER_SIZE:
            *value = 32;
            break;
        case EGL_RED_SIZE:
        case EGL_GREEN_SIZE:
        case EGL_BLUE_SIZE:
        case EGL_ALPHA_SIZE:
            *value = 8;
            break;
        case EGL_DEPTH_SIZE:
        case EGL_STENCIL_SIZE:
        case EGL_SAMPLES:
        case EGL_SAMPLE_BUFFERS:
        case EGL_LUMINANCE_SIZE:
        case EGL_ALPHA_MASK_SIZE:
        case EGL_LEVEL:
        case EGL_NATIVE_VISUAL_ID:
        case EGL_NATIVE_VISUAL_TYPE:
        case EGL_TRANSPARENT_RED_VALUE:
        case EGL_TRANSPARENT_GREEN_VALUE:
        case EGL_TRANSPARENT_BLUE_VALUE:
        case EGL_MAX_PBUFFER_WIDTH:
        case EGL_MAX_PBUFFER_HEIGHT:
        case EGL_MAX_PBUFFER_PIXELS:
        case EGL_MIN_SWAP_INTERVAL:
            *value = 0;
            break;
        case EGL_MAX_SWAP_INTERVAL:
            *value = 1;
            break;
        case EGL_CONFIG_ID:
            *value = 1;
            break;
        case EGL_CONFIG_CAVEAT:
            *value = EGL_NONE;
            break;
        case EGL_COLOR_BUFFER_TYPE:
            *value = EGL_RGB_BUFFER;
            break;
        case EGL_COLOR_COMPONENT_TYPE_EXT:
            *value = EGL_COLOR_COMPONENT_TYPE_FIXED_EXT;
            break;
        case EGL_NATIVE_RENDERABLE:
            *value = EGL_TRUE;
            break;
        case EGL_BIND_TO_TEXTURE_RGB:
        case EGL_BIND_TO_TEXTURE_RGBA:
            *value = EGL_FALSE;
            break;
        case EGL_SURFACE_TYPE:
            *value = EGL_WINDOW_BIT | EGL_PBUFFER_BIT;
            break;
        case EGL_RENDERABLE_TYPE:
        case EGL_CONFORMANT:
            *value = EGL_OPENGL_BIT;
            break;
        case EGL_TRANSPARENT_TYPE:
            *value = EGL_NONE;
            break;
        default:
            return SetError(EGL_BAD_ATTRIBUTE);
    }
    return SetError(EGL_SUCCESS);
}

EGLBoolean EGLAPIENTRY GetConfigs(EGLDisplay dpy,
                                  EGLConfig* configs,
                                  EGLint configSize,
                                  EGLint* numConfig) {
    if (!IsValidDisplay(dpy)) {
        return EGL_FALSE;
    }
    if (numConfig == nullptr) {
        return SetError(EGL_BAD_PARAMETER);
    }
    *numConfig = 1;
    if (configs != nullptr && configSize >= 1) {
        configs[0] = kConfig;
    }
    return SetError(EGL_SUCCESS);
}

EGLDisplay EGLAPIENTRY GetCurrentDisplay() {
    return tState.context != nullptr ? kDisplay : EGL_NO_DISPLAY;
}

EGLSurface EGLAPIENTRY GetCurrentSurface(EGLint readdraw) {
    switch (readdraw) {
        case EGL_DRAW:
            return tState.draw != nullptr ? tState.draw : EGL_NO_SURFACE;
        case EGL_READ:
            return tState.read != nullptr ? tState.read : EGL_NO_SURFACE;
        default:
            SetError(EGL_BAD_PARAMETER);
            return EGL_NO_SURFACE;
    }
}

EGLDisplay EGLAPIENTRY GetDisplay(EGLNativeDisplayType displayId) {
    return displayId == EGL_DEFAULT_DISPLAY ? kDisplay : EGL_NO_DISPLAY;
}

EGLint EGLAPIENTRY GetError() {
    EGLint error = tState.error;
    tState.error = EGL_SUCCESS;
    return error;
}

EGLBoolean EGLAPIENTRY Initialize(EGLDisplay dpy, EGLint* major, EGLint* minor) {
    if (dpy != kDisplay) {
        return SetError(EGL_BAD_DISPLAY);
    }
    Display& d = GetDisplayState();
    {
        std::lock_guard<std::mutex> lock(d.mutex);
        if (!d.initialized) {
            if (!InitializeDisplay(d)) {
                return SetError(EGL_NOT_INITIALIZED);
            }
            d.initialized = true;
        }
    }
    if (major != nullptr) {
        *major = 1;
    }
    if (minor != nullptr) {
        *minor = 4;
    }
    return SetError(EGL_SUCCESS);
}

EGLBoolean EGLAPIENTRY MakeCurrent(EGLDisplay dpy,
                                   EGLSurface draw,
                                   EGLSurface read,
                                   EGLContext ctx) {
    if (!IsValidDisplay(dpy)) {
        return EGL_FALSE;
    }
    Display& d = GetDisplayState();
    Context* context = static_cast<Context*>(ctx);
    Surface* drawSurface = static_cast<Surface*>(draw);
    Surface* readSurface = static_cast<Surface*>(read);
    if (context == nullptr) {
        if (drawSurface != nullptr || readSurface != nullptr) {
            return SetError(EGL_BAD_MATCH);
        }
        d.makeCurrent(nullptr, nullptr);
        tState = {nullptr, nullptr, nullptr, EGL_SUCCESS};
        return EGL_TRUE;
    }
    // WGL draws and reads through one DC.
    if (drawSurface != readSurface) {
        return SetError(EGL_BAD_MATCH);
    }
    HDC dc = drawSurface != nullptr ? drawSurface->dc : SurfacelessDC(d);
    if (DWORD error = MakeCurrentLogged(d, dc, context->rc); error != 0) {
        return SetError(error == ERROR_BUSY ? EGL_BAD_ACCESS : EGL_BAD_MATCH);
    }
    tState.context = context;
    tState.draw = drawSurface;
    tState.read = readSurface;
    LoadGLProcsOnce();
    FlushPendingDeletes(d);
    return SetError(EGL_SUCCESS);
}

EGLBoolean EGLAPIENTRY QueryContext(EGLDisplay dpy, EGLContext ctx, EGLint attribute, EGLint* value) {
    if (!IsValidDisplay(dpy)) {
        return EGL_FALSE;
    }
    if (ctx == EGL_NO_CONTEXT) {
        return SetError(EGL_BAD_CONTEXT);
    }
    if (value == nullptr) {
        return SetError(EGL_BAD_PARAMETER);
    }
    switch (attribute) {
        case EGL_CONFIG_ID:
            *value = 1;
            break;
        case EGL_CONTEXT_CLIENT_TYPE:
            *value = EGL_OPENGL_API;
            break;
        case EGL_RENDER_BUFFER:
            *value = tState.context == ctx && tState.draw == nullptr ? EGL_NONE : EGL_BACK_BUFFER;
            break;
        default:
            return SetError(EGL_BAD_ATTRIBUTE);
    }
    return SetError(EGL_SUCCESS);
}

const char* EGLAPIENTRY QueryString(EGLDisplay dpy, EGLint name) {
    // No client extensions: EGL_NO_DISPLAY queries fail like on a plain EGL 1.4.
    if (!IsValidDisplay(dpy)) {
        return nullptr;
    }
    SetError(EGL_SUCCESS);
    switch (name) {
        case EGL_CLIENT_APIS:
            return "OpenGL";
        case EGL_EXTENSIONS:
            return kDisplayExtensions;
        case EGL_VENDOR:
            return "Dawn (EGL on WGL)";
        case EGL_VERSION:
            return "1.4 WGL";
        default:
            SetError(EGL_BAD_PARAMETER);
            return nullptr;
    }
}

EGLBoolean EGLAPIENTRY QuerySurface(EGLDisplay dpy, EGLSurface surf, EGLint attribute, EGLint* value) {
    if (!IsValidDisplay(dpy)) {
        return EGL_FALSE;
    }
    if (surf == EGL_NO_SURFACE) {
        return SetError(EGL_BAD_SURFACE);
    }
    if (value == nullptr) {
        return SetError(EGL_BAD_PARAMETER);
    }
    Surface* surface = static_cast<Surface*>(surf);
    switch (attribute) {
        case EGL_WIDTH:
        case EGL_HEIGHT: {
            RECT rect = {};
            if (!::GetClientRect(surface->hwnd, &rect)) {
                return SetError(EGL_BAD_NATIVE_WINDOW);
            }
            *value = attribute == EGL_WIDTH ? rect.right - rect.left : rect.bottom - rect.top;
            break;
        }
        case EGL_CONFIG_ID:
            *value = 1;
            break;
        case EGL_RENDER_BUFFER:
            *value = EGL_BACK_BUFFER;
            break;
        case EGL_SWAP_BEHAVIOR:
            *value = EGL_BUFFER_DESTROYED;
            break;
        default:
            return SetError(EGL_BAD_ATTRIBUTE);
    }
    return SetError(EGL_SUCCESS);
}

EGLBoolean EGLAPIENTRY SwapBuffersImpl(EGLDisplay dpy, EGLSurface surf) {
    if (!IsValidDisplay(dpy)) {
        return EGL_FALSE;
    }
    Surface* surface = static_cast<Surface*>(surf);
    if (surface == nullptr || tState.draw != surface) {
        return SetError(EGL_BAD_SURFACE);
    }
    Display& d = GetDisplayState();
    int interval = d.desiredSwapInterval.load();
    if (surface->appliedSwapInterval != interval && d.swapInterval != nullptr) {
        d.swapInterval(interval);
        surface->appliedSwapInterval = interval;
    }
    if (!::SwapBuffers(surface->dc)) {
        return SetError(EGL_BAD_SURFACE);
    }
    return SetError(EGL_SUCCESS);
}

EGLBoolean EGLAPIENTRY Terminate(EGLDisplay dpy) {
    // The display is process-wide and kept alive, like an EGLDisplay shared between users.
    return dpy == kDisplay ? SetError(EGL_SUCCESS) : SetError(EGL_BAD_DISPLAY);
}

EGLBoolean EGLAPIENTRY WaitGL() {
    return SetError(EGL_SUCCESS);
}

EGLBoolean EGLAPIENTRY WaitNative(EGLint) {
    return SetError(EGL_SUCCESS);
}

// EGL 1.1

EGLBoolean EGLAPIENTRY BindTexImage(EGLDisplay, EGLSurface, EGLint) {
    return SetError(EGL_BAD_MATCH);
}

EGLBoolean EGLAPIENTRY ReleaseTexImage(EGLDisplay, EGLSurface, EGLint) {
    return SetError(EGL_BAD_MATCH);
}

EGLBoolean EGLAPIENTRY SurfaceAttrib(EGLDisplay dpy, EGLSurface, EGLint, EGLint) {
    if (!IsValidDisplay(dpy)) {
        return EGL_FALSE;
    }
    return SetError(EGL_BAD_ATTRIBUTE);
}

EGLBoolean EGLAPIENTRY SwapInterval(EGLDisplay dpy, EGLint interval) {
    if (!IsValidDisplay(dpy)) {
        return EGL_FALSE;
    }
    GetDisplayState().desiredSwapInterval = interval < 0 ? 0 : (interval > 1 ? 1 : interval);
    return SetError(EGL_SUCCESS);
}

// EGL 1.2

EGLBoolean EGLAPIENTRY BindAPI(EGLenum api) {
    return api == EGL_OPENGL_API ? SetError(EGL_SUCCESS) : SetError(EGL_BAD_PARAMETER);
}

EGLenum EGLAPIENTRY QueryAPI() {
    return EGL_OPENGL_API;
}

EGLSurface EGLAPIENTRY
CreatePbufferFromClientBuffer(EGLDisplay, EGLenum, EGLClientBuffer, EGLConfig, const EGLint*) {
    SetError(EGL_BAD_PARAMETER);
    return EGL_NO_SURFACE;
}

EGLBoolean EGLAPIENTRY ReleaseThread() {
    if (tState.context != nullptr) {
        GetDisplayState().makeCurrent(nullptr, nullptr);
    }
    tState = {};
    return EGL_TRUE;
}

EGLBoolean EGLAPIENTRY WaitClient() {
    return SetError(EGL_SUCCESS);
}

// EGL 1.4

EGLContext EGLAPIENTRY GetCurrentContext() {
    return tState.context != nullptr ? tState.context : EGL_NO_CONTEXT;
}

// EGL_KHR_fence_sync, on GL sync objects: they live in the context's share group, and Dawn
// creates and waits on them with its context current.

EGLSyncKHR EGLAPIENTRY CreateSyncKHR(EGLDisplay dpy, EGLenum type, const EGLint* attribs) {
    if (!IsValidDisplay(dpy)) {
        return EGL_NO_SYNC_KHR;
    }
    if (type != EGL_SYNC_FENCE_KHR || (attribs != nullptr && attribs[0] != EGL_NONE)) {
        SetError(type != EGL_SYNC_FENCE_KHR ? EGL_BAD_PARAMETER : EGL_BAD_ATTRIBUTE);
        return EGL_NO_SYNC_KHR;
    }
    Display& d = GetDisplayState();
    if (tState.context == nullptr || d.fenceSync == nullptr) {
        SetError(EGL_BAD_MATCH);
        return EGL_NO_SYNC_KHR;
    }
    GLsyncHandle sync = d.fenceSync(kGL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    if (sync == nullptr) {
        SetError(EGL_BAD_ALLOC);
        return EGL_NO_SYNC_KHR;
    }
    SetError(EGL_SUCCESS);
    return static_cast<EGLSyncKHR>(sync);
}

EGLBoolean EGLAPIENTRY DestroySyncKHR(EGLDisplay dpy, EGLSyncKHR sync) {
    if (!IsValidDisplay(dpy)) {
        return EGL_FALSE;
    }
    if (sync == EGL_NO_SYNC_KHR) {
        return SetError(EGL_BAD_PARAMETER);
    }
    Display& d = GetDisplayState();
    if (tState.context != nullptr && d.deleteSync != nullptr) {
        d.deleteSync(static_cast<GLsyncHandle>(sync));
    } else {
        std::lock_guard<std::mutex> lock(d.mutex);
        d.pendingDeletes.push_back(static_cast<GLsyncHandle>(sync));
    }
    return SetError(EGL_SUCCESS);
}

EGLint EGLAPIENTRY ClientWaitSyncKHR(EGLDisplay dpy,
                                     EGLSyncKHR sync,
                                     EGLint flags,
                                     EGLTimeKHR timeout) {
    if (!IsValidDisplay(dpy)) {
        return EGL_FALSE;
    }
    Display& d = GetDisplayState();
    if (sync == EGL_NO_SYNC_KHR) {
        return SetError(EGL_BAD_PARAMETER);
    }
    if (tState.context == nullptr || d.clientWaitSync == nullptr) {
        return SetError(EGL_BAD_MATCH);
    }
    // EGL_FOREVER_KHR and GL_TIMEOUT_IGNORED are both ~0ull.
    unsigned glFlags = (flags & EGL_SYNC_FLUSH_COMMANDS_BIT_KHR) ? kGL_SYNC_FLUSH_COMMANDS_BIT : 0;
    unsigned result = d.clientWaitSync(static_cast<GLsyncHandle>(sync), glFlags, timeout);
    switch (result) {
        case kGL_ALREADY_SIGNALED:
        case kGL_CONDITION_SATISFIED:
            SetError(EGL_SUCCESS);
            return EGL_CONDITION_SATISFIED_KHR;
        case kGL_TIMEOUT_EXPIRED:
            SetError(EGL_SUCCESS);
            return EGL_TIMEOUT_EXPIRED_KHR;
        default:
            return SetError(EGL_BAD_ACCESS);
    }
}

EGLBoolean EGLAPIENTRY GetSyncAttribKHR(EGLDisplay dpy,
                                        EGLSyncKHR sync,
                                        EGLint attribute,
                                        EGLint* value) {
    if (!IsValidDisplay(dpy)) {
        return EGL_FALSE;
    }
    if (sync == EGL_NO_SYNC_KHR || value == nullptr) {
        return SetError(EGL_BAD_PARAMETER);
    }
    switch (attribute) {
        case EGL_SYNC_TYPE_KHR:
            *value = EGL_SYNC_FENCE_KHR;
            break;
        case EGL_SYNC_CONDITION_KHR:
            *value = EGL_SYNC_PRIOR_COMMANDS_COMPLETE_KHR;
            break;
        case EGL_SYNC_STATUS_KHR: {
            Display& d = GetDisplayState();
            if (tState.context == nullptr || d.getSynciv == nullptr) {
                return SetError(EGL_BAD_MATCH);
            }
            int status = 0;
            d.getSynciv(static_cast<GLsyncHandle>(sync), kGL_SYNC_STATUS, 1, nullptr, &status);
            *value = status == kGL_SIGNALED ? EGL_SIGNALED_KHR : EGL_UNSIGNALED_KHR;
            break;
        }
        default:
            return SetError(EGL_BAD_ATTRIBUTE);
    }
    return SetError(EGL_SUCCESS);
}

struct ProcEntry {
    const char* name;
    __eglMustCastToProperFunctionPointerType proc;
};

#define EGL_PROC(name, fn) {name, reinterpret_cast<__eglMustCastToProperFunctionPointerType>(&fn)}

const ProcEntry kProcs[] = {
    EGL_PROC("eglChooseConfig", ChooseConfig),
    EGL_PROC("eglCopyBuffers", CopyBuffers),
    EGL_PROC("eglCreateContext", CreateContext),
    EGL_PROC("eglCreatePbufferSurface", CreatePbufferSurface),
    EGL_PROC("eglCreatePixmapSurface", CreatePixmapSurface),
    EGL_PROC("eglCreateWindowSurface", CreateWindowSurface),
    EGL_PROC("eglDestroyContext", DestroyContext),
    EGL_PROC("eglDestroySurface", DestroySurface),
    EGL_PROC("eglGetConfigAttrib", GetConfigAttrib),
    EGL_PROC("eglGetConfigs", GetConfigs),
    EGL_PROC("eglGetCurrentDisplay", GetCurrentDisplay),
    EGL_PROC("eglGetCurrentSurface", GetCurrentSurface),
    EGL_PROC("eglGetDisplay", GetDisplay),
    EGL_PROC("eglGetError", GetError),
    EGL_PROC("eglInitialize", Initialize),
    EGL_PROC("eglMakeCurrent", MakeCurrent),
    EGL_PROC("eglQueryContext", QueryContext),
    EGL_PROC("eglQueryString", QueryString),
    EGL_PROC("eglQuerySurface", QuerySurface),
    EGL_PROC("eglSwapBuffers", SwapBuffersImpl),
    EGL_PROC("eglTerminate", Terminate),
    EGL_PROC("eglWaitGL", WaitGL),
    EGL_PROC("eglWaitNative", WaitNative),
    EGL_PROC("eglBindTexImage", BindTexImage),
    EGL_PROC("eglReleaseTexImage", ReleaseTexImage),
    EGL_PROC("eglSurfaceAttrib", SurfaceAttrib),
    EGL_PROC("eglSwapInterval", SwapInterval),
    EGL_PROC("eglBindAPI", BindAPI),
    EGL_PROC("eglQueryAPI", QueryAPI),
    EGL_PROC("eglCreatePbufferFromClientBuffer", CreatePbufferFromClientBuffer),
    EGL_PROC("eglReleaseThread", ReleaseThread),
    EGL_PROC("eglWaitClient", WaitClient),
    EGL_PROC("eglGetCurrentContext", GetCurrentContext),
    EGL_PROC("eglCreateSyncKHR", CreateSyncKHR),
    EGL_PROC("eglDestroySyncKHR", DestroySyncKHR),
    EGL_PROC("eglClientWaitSyncKHR", ClientWaitSyncKHR),
    EGL_PROC("eglGetSyncAttribKHR", GetSyncAttribKHR),
};

#undef EGL_PROC

__eglMustCastToProperFunctionPointerType EGLAPIENTRY GetProcAddressImpl(const char* name) {
    if (name == nullptr) {
        return nullptr;
    }
    if (std::strncmp(name, "egl", 3) == 0) {
        for (const ProcEntry& entry : kProcs) {
            if (std::strcmp(entry.name, name) == 0) {
                return entry.proc;
            }
        }
        return nullptr;
    }
    return reinterpret_cast<__eglMustCastToProperFunctionPointerType>(LookupGLProc(name));
}

}  // anonymous namespace

EGLGetProcProc GetEGLOnWGLGetProc() {
    return &GetProcAddressImpl;
}

}  // namespace dawn::native::opengl
