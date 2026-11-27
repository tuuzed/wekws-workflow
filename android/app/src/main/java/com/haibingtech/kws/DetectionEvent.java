package com.haibingtech.kws;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/** One keyword detection returned by the native engine. */
public final class DetectionEvent {
    public final int channel;
    public final String keyword;
    public final float score;
    public final int startMs;
    public final int endMs;
    public final List<TokenProb> tokenProbs;

    public DetectionEvent(int channel, String keyword, float score,
                          int startMs, int endMs,
                          List<TokenProb> tokenProbs) {
        this.channel = channel;
        this.keyword = keyword;
        this.score = score;
        this.startMs = startMs;
        this.endMs = endMs;
        this.tokenProbs = tokenProbs == null
                ? new ArrayList<TokenProb>() : tokenProbs;
    }

    static List<DetectionEvent> fromJson(String json) throws JSONException {
        List<DetectionEvent> events = new ArrayList<>();
        JSONArray array = new JSONArray(json);
        for (int i = 0; i < array.length(); ++i) {
            JSONObject item = array.getJSONObject(i);
            events.add(new DetectionEvent(
                    item.getInt("channel"),
                    item.getString("keyword"),
                    (float) item.getDouble("score"),
                    item.getInt("start_ms"),
                    item.getInt("end_ms"),
                    TokenProb.fromJson(item.optJSONArray("token_probs"))));
        }
        return events;
    }

    /** Multi-line representation used by the Android Events panel. */
    public String formatForEventsPanel() {
        StringBuilder text = new StringBuilder();
        text.append("channel=").append(channel)
                .append("  keyword=").append(keyword)
                .append("\nscore=")
                .append(String.format(Locale.US, "%.6f", score))
                .append("  start_ms=").append(startMs)
                .append("  end_ms=").append(endMs)
                .append("\ntoken_probs:");
        for (TokenProb token : tokenProbs) {
            text.append("\n  ").append(token.formatForEventsPanel());
        }
        return text.toString();
    }

    @Override
    public String toString() {
        return "channel=" + channel
                + " keyword=" + keyword
                + " score=" + String.format("%.6f", score)
                + " start_ms=" + startMs
                + " end_ms=" + endMs
                + " tokens=" + tokenProbs;
    }
}
