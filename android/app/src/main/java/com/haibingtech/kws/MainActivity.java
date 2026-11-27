package com.haibingtech.kws;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.TextView;

import org.json.JSONException;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Minimal microphone demo for the multi-channel native engine.
 *
 * <p>The UI feeds the selected channel only, so switching the spinner shows
 * that each channel owns independent frontend/cache/decoder state. Real
 * callers can feed several independent PCM streams by calling
 * {@link KwsEngine#feed(int, short[], int, int)} for each channel.</p>
 */
public class MainActivity extends Activity {
    private static final int REQUEST_RECORD_AUDIO = 1;
    private static final int SAMPLE_RATE = 16000;
    private static final int CHUNK_SAMPLES = 1600; // 100 ms
    private static final long EVENT_DRAIN_INTERVAL_MS = 100;

    private static final class LiveEvent {
        DetectionEvent event;
        boolean finalEvent;

        LiveEvent(DetectionEvent event, boolean finalEvent) {
            this.event = event;
            this.finalEvent = finalEvent;
        }
    }

    private final AtomicBoolean running = new AtomicBoolean(false);
    // Key: channel/keyword/startMs/endMs. One row per event signature; the
    // score is refreshed in place instead of appending every decode frame.
    private final Map<String, LiveEvent> liveEvents = new LinkedHashMap<>();
    private long lastEventDrainMs = 0;
    private final Handler uiHandler = new Handler(Looper.getMainLooper());

    private KwsEngine engine;
    private AudioRecord recorder;
    private Thread recordThread;
    private Thread decodeThread;
    private TextView statusView;
    private TextView eventView;
    private TextView logView;
    private Spinner channelSpinner;
    private Button startButton;
    private Button stopButton;
    private Button debugButton;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        statusView = findViewById(R.id.statusView);
        eventView = findViewById(R.id.eventView);
        logView = findViewById(R.id.logView);
        channelSpinner = findViewById(R.id.channelSpinner);
        startButton = findViewById(R.id.startButton);
        stopButton = findViewById(R.id.stopButton);
        debugButton = findViewById(R.id.debugButton);
        stopButton.setEnabled(false);

        try {
            // channels <= 0 means "read runtime.default_channels from config.json".
            engine = new KwsEngine(this, "kws.onnx", "config.json", 0);
            List<String> channelNames = new ArrayList<>();
            for (int c = 0; c < engine.channels(); ++c) {
                channelNames.add("channel " + c);
            }
            ArrayAdapter<String> adapter = new ArrayAdapter<>(this,
                    android.R.layout.simple_spinner_item, channelNames);
            adapter.setDropDownViewResource(
                    android.R.layout.simple_spinner_dropdown_item);
            channelSpinner.setAdapter(adapter);
            setStatus("Ready, " + engine.channels() + " channel(s)");
        } catch (Exception e) {
            setStatus("Engine init failed: " + e.getMessage());
            startButton.setEnabled(false);
            debugButton.setEnabled(false);
        }

        startButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (checkSelfPermission(Manifest.permission.RECORD_AUDIO)
                        != PackageManager.PERMISSION_GRANTED) {
                    requestPermissions(
                            new String[]{Manifest.permission.RECORD_AUDIO},
                            REQUEST_RECORD_AUDIO);
                } else {
                    startRecording();
                }
            }
        });

        stopButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                stopRecording();
            }
        });

        debugButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                dumpFrameDebug();
            }
        });
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions,
                                           int[] grantResults) {
        if (requestCode == REQUEST_RECORD_AUDIO
                && grantResults.length > 0
                && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startRecording();
        } else {
            setStatus("RECORD_AUDIO permission denied");
        }
    }

    private void startRecording() {
        if (engine == null || !running.compareAndSet(false, true)) {
            return;
        }
        synchronized (liveEvents) {
            liveEvents.clear();
        }
        lastEventDrainMs = 0;
        renderEvents();
        for (int c = 0; c < engine.channels(); ++c) {
            engine.resetChannel(c);
        }

        final int channel = channelSpinner.getSelectedItemPosition();
        recordThread = new Thread(new Runnable() {
            @Override
            public void run() {
                recordLoop(channel);
            }
        }, "kws-record");
        decodeThread = new Thread(new Runnable() {
            @Override
            public void run() {
                decodeLoop();
            }
        }, "kws-decode");
        recordThread.start();
        decodeThread.start();

        startButton.setEnabled(false);
        stopButton.setEnabled(true);
        setStatus("Listening on channel " + channel);
    }

    private void stopRecording() {
        running.set(false);
        joinThread(recordThread);
        joinThread(decodeThread);
        recordThread = null;
        decodeThread = null;

        if (engine != null) {
            for (int c = 0; c < engine.channels(); ++c) {
                engine.finish(c);
            }
            try {
                // Finish() first so the final beam is flushed; then mark every
                // row final before ResetChannel() clears native event state.
                drainEvents(true);
            } catch (JSONException e) {
                appendLog("drain error: " + e.getMessage());
            }
            for (int c = 0; c < engine.channels(); ++c) {
                engine.resetChannel(c);
            }
        }
        startButton.setEnabled(true);
        stopButton.setEnabled(false);
        setStatus("Stopped");
    }

    private void recordLoop(int channel) {
        try {
            int minBuffer = AudioRecord.getMinBufferSize(SAMPLE_RATE,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT);
            if (minBuffer <= 0) {
                throw new IllegalStateException("AudioRecord.getMinBufferSize failed");
            }
            int bufferSize = Math.max(minBuffer, CHUNK_SAMPLES * 2 * 2);
            recorder = new AudioRecord(MediaRecorder.AudioSource.MIC, SAMPLE_RATE,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT, bufferSize);
            if (recorder.getState() != AudioRecord.STATE_INITIALIZED) {
                throw new IllegalStateException("AudioRecord init failed");
            }
            recorder.startRecording();
            short[] pcm = new short[CHUNK_SAMPLES];
            while (running.get()) {
                int n = recorder.read(pcm, 0, pcm.length);
                if (n > 0) {
                    engine.feed(channel, pcm, 0, n);
                }
            }
        } catch (Exception e) {
            appendLog("record error: " + e.getMessage());
        } finally {
            if (recorder != null) {
                try {
                    recorder.stop();
                } catch (IllegalStateException ignored) {
                    // Recording may not have started yet.
                }
                recorder.release();
                recorder = null;
            }
        }
    }

    private void decodeLoop() {
        while (running.get()) {
            try {
                // Keep decoding during capture and pull events periodically so
                // the UI shows live hits without waiting for Stop. The score is
                // still refined until Finish() flushes the final beam.
                boolean decoded = engine.decodeOnce();
                long now = System.currentTimeMillis();
                if (now - lastEventDrainMs >= EVENT_DRAIN_INTERVAL_MS) {
                    try {
                        drainEvents(false);
                        lastEventDrainMs = now;
                    } catch (JSONException e) {
                        appendLog("drain error: " + e.getMessage());
                    }
                }
                if (!decoded) {
                    Thread.sleep(5);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                appendLog("decode error: " + e.getMessage());
                break;
            }
        }
    }

    private void drainEvents(final boolean finalFlush) throws JSONException {
        for (DetectionEvent event : engine.currentEvents()) {
            String signature = event.channel + "/" + event.keyword + "/"
                    + event.startMs + "/" + event.endMs;
            synchronized (liveEvents) {
                LiveEvent previous = liveEvents.get(signature);
                if (previous == null) {
                    liveEvents.put(signature, new LiveEvent(event, finalFlush));
                } else {
                    // Same event, possibly refined: refresh the score in place.
                    previous.event = event;
                    if (finalFlush) {
                        previous.finalEvent = true;
                    }
                }
            }
        }
        if (finalFlush) {
            synchronized (liveEvents) {
                for (LiveEvent item : liveEvents.values()) {
                    item.finalEvent = true;
                }
            }
        }
        renderEvents();
    }

    private void renderEvents() {
        final StringBuilder text = new StringBuilder();
        synchronized (liveEvents) {
            text.append("Events (").append(liveEvents.size()).append("):");
            for (Map.Entry<String, LiveEvent> item : liveEvents.entrySet()) {
                DetectionEvent event = item.getValue().event;
                text.append("\n")
                        .append(item.getValue().finalEvent ? "final" : "live")
                        .append(" ch=").append(event.channel)
                        .append(" ").append(event.keyword)
                        .append(" score=")
                        .append(String.format(Locale.US, "%.6f", event.score))
                        .append(" ").append(event.startMs).append("-")
                        .append(event.endMs).append("ms");
            }
        }
        uiHandler.post(new Runnable() {
            @Override
            public void run() {
                eventView.setText(text.toString());
            }
        });
    }

    private void dumpFrameDebug() {
        if (engine == null) {
            return;
        }
        int channel = Math.max(0, channelSpinner.getSelectedItemPosition());
        try {
            List<FrameDebug> frames = engine.frameDebug(channel, 200);
            appendLog("frame debug channel=" + channel
                    + " frames=" + frames.size());
            for (FrameDebug frame : frames) {
                appendLog(frame.toString());
            }
        } catch (Exception e) {
            appendLog("debug error: " + e.getMessage());
        }
    }

    private static void joinThread(Thread thread) {
        if (thread == null) {
            return;
        }
        try {
            thread.join(2000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    private void setStatus(final String text) {
        uiHandler.post(new Runnable() {
            @Override
            public void run() {
                statusView.setText(text);
            }
        });
    }

    private void appendLog(final String text) {
        uiHandler.post(new Runnable() {
            @Override
            public void run() {
                logView.append(text + "\n");
            }
        });
    }

    @Override
    protected void onDestroy() {
        stopRecording();
        if (engine != null) {
            engine.close();
            engine = null;
        }
        super.onDestroy();
    }
}
