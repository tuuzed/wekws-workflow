package com.haibingtech.kws;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** Per-frame top-k posterior debug data for one channel. */
public final class FrameDebug {
    public final int frame;
    public final int timeMs;
    public final List<FrameTokenProb> rawTopk;
    public final List<FrameTokenProb> kept;

    public FrameDebug(int frame, int timeMs,
                      List<FrameTokenProb> rawTopk,
                      List<FrameTokenProb> kept) {
        this.frame = frame;
        this.timeMs = timeMs;
        this.rawTopk = rawTopk;
        this.kept = kept;
    }

    static List<FrameDebug> fromJson(String json,
                                     Map<Integer, String> idToToken)
            throws JSONException {
        List<FrameDebug> result = new ArrayList<>();
        JSONArray array = new JSONArray(json);
        for (int i = 0; i < array.length(); ++i) {
            JSONObject item = array.getJSONObject(i);
            result.add(new FrameDebug(
                    item.getInt("frame"),
                    item.getInt("time_ms"),
                    FrameTokenProb.fromJson(item.optJSONArray("raw_topk"), idToToken),
                    FrameTokenProb.fromJson(item.optJSONArray("kept"), idToToken)));
        }
        return result;
    }

    @Override
    public String toString() {
        return "frame=" + frame
                + " ms=" + timeMs
                + " raw=" + rawTopk
                + " kept=" + kept;
    }
}
