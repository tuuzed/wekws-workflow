package com.haibingtech.kws;

import android.content.Context;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * Multi-channel streaming KWS engine.
 *
 * <p>The model and configuration are read from assets. All audio state is kept
 * in native code; call {@link #feed(int, short[], int, int)} from the capture
 * thread and {@link #decodeOnce()} from a single decode thread.</p>
 */
public final class KwsEngine implements AutoCloseable {
    private final int channels;
    private final Map<Integer, String> idToToken = new HashMap<>();
    private long handle;

    public KwsEngine(Context context, String modelAsset, String configAsset,
                     int channels) throws IOException, JSONException {
        String configJson = readAsset(context, configAsset);
        int resolvedChannels = channels;
        if (resolvedChannels <= 0) {
            JSONObject runtime = new JSONObject(configJson).optJSONObject("runtime");
            resolvedChannels = runtime == null ? 1 : runtime.optInt("default_channels", 1);
        }
        if (resolvedChannels <= 0) {
            throw new IllegalArgumentException("channels must be positive");
        }

        try {
            JSONObject tokenMap = new JSONObject(readAsset(context, "id_to_token.json"));
            Iterator<String> keys = tokenMap.keys();
            while (keys.hasNext()) {
                String key = keys.next();
                idToToken.put(Integer.parseInt(key), tokenMap.getString(key));
            }
        } catch (Exception ignored) {
            // Debug-only map; frame debug falls back to "<id:n>" labels.
        }

        File modelFile = copyAssetToFiles(context, modelAsset);
        this.channels = resolvedChannels;
        this.handle = KwsJni.create(modelFile.getAbsolutePath(), configJson,
                resolvedChannels);
        if (this.handle == 0) {
            throw new IOException("failed to create native KWS engine");
        }
    }

    public int channels() {
        return channels;
    }

    public void feed(int channel, short[] pcm, int offset, int length) {
        ensureOpen();
        KwsJni.feed(handle, channel, pcm, offset, length);
    }

    public void finish(int channel) {
        ensureOpen();
        KwsJni.finish(handle, channel);
    }

    /** Runs one batched ONNX inference step. Returns false when drained. */
    public boolean decodeOnce() {
        ensureOpen();
        return KwsJni.decodeOnce(handle);
    }

    /** Best hit currently held for each channel, above the trigger threshold. */
    public List<DetectionEvent> currentEvents() throws JSONException {
        ensureOpen();
        return DetectionEvent.fromJson(KwsJni.currentEventsJson(handle));
    }

    /** Per-frame top-k debug data. maxFrames <= 0 returns the full buffer. */
    public List<FrameDebug> frameDebug(int channel, int maxFrames)
            throws JSONException {
        ensureOpen();
        return FrameDebug.fromJson(
                KwsJni.frameDebugJson(handle, channel, maxFrames), idToToken);
    }

    public void resetChannel(int channel) {
        ensureOpen();
        KwsJni.resetChannel(handle, channel);
    }

    @Override
    public synchronized void close() {
        if (handle != 0) {
            KwsJni.destroy(handle);
            handle = 0;
        }
    }

    private void ensureOpen() {
        if (handle == 0) {
            throw new IllegalStateException("KwsEngine is closed");
        }
    }

    private static String readAsset(Context context, String name)
            throws IOException {
        StringBuilder builder = new StringBuilder();
        char[] buffer = new char[4096];
        try (Reader reader = new InputStreamReader(
                context.getAssets().open(name), StandardCharsets.UTF_8)) {
            int n;
            while ((n = reader.read(buffer)) != -1) {
                builder.append(buffer, 0, n);
            }
        }
        return builder.toString();
    }

    private static File copyAssetToFiles(Context context, String name)
            throws IOException {
        File output = new File(context.getFilesDir(), new File(name).getName());
        try (InputStream input = context.getAssets().open(name);
             FileOutputStream fos = new FileOutputStream(output)) {
            byte[] buffer = new byte[64 * 1024];
            int n;
            while ((n = input.read(buffer)) != -1) {
                fos.write(buffer, 0, n);
            }
        }
        return output;
    }
}
