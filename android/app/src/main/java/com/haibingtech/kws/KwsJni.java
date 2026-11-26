package com.haibingtech.kws;

/** Thin JNI wrapper. Do not call these methods directly in application code. */
final class KwsJni {
    static {
        System.loadLibrary("kws");
    }

    private KwsJni() {
    }

    static native long create(String modelPath, String configJson, int channels);

    static native void feed(long handle, int channel, short[] pcm,
                            int offset, int length);

    static native void finish(long handle, int channel);

    static native boolean decodeOnce(long handle);

    static native String currentEventsJson(long handle);

    static native String frameDebugJson(long handle, int channel, int maxFrames);

    static native void resetChannel(long handle, int channel);

    static native void destroy(long handle);
}
