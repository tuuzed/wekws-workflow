package com.haibingtech.kws;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/** One token probability from the winning CTC keyword path. */
public final class TokenProb {
    public final String token;
    public final int tokenId;
    public final int frame;
    public final float prob;

    public TokenProb(String token, int tokenId, int frame, float prob) {
        this.token = token;
        this.tokenId = tokenId;
        this.frame = frame;
        this.prob = prob;
    }

    static List<TokenProb> fromJson(JSONArray array) throws JSONException {
        List<TokenProb> result = new ArrayList<>();
        if (array == null) {
            return result;
        }
        for (int i = 0; i < array.length(); ++i) {
            JSONObject item = array.getJSONObject(i);
            result.add(new TokenProb(
                    item.optString("token", ""),
                    item.getInt("token_id"),
                    item.getInt("frame"),
                    (float) item.getDouble("prob")));
        }
        return result;
    }

    /** Single-line representation used by the Android Events panel. */
    public String formatForEventsPanel() {
        return String.format(Locale.US,
                "token=%s  token_id=%d  frame=%d  prob=%.6f",
                token, tokenId, frame, prob);
    }

    @Override
    public String toString() {
        return token + "=" + String.format("%.6f", prob) + "@" + frame;
    }
}
