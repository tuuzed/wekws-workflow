package com.haibingtech.kws;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** One top-k token entry from a debug frame. */
public final class FrameTokenProb {
    public final int tokenId;
    public final String token;
    public final float prob;

    public FrameTokenProb(int tokenId, String token, float prob) {
        this.tokenId = tokenId;
        this.token = token;
        this.prob = prob;
    }

    static List<FrameTokenProb> fromJson(JSONArray array,
                                         Map<Integer, String> idToToken)
            throws JSONException {
        List<FrameTokenProb> result = new ArrayList<>();
        if (array == null) {
            return result;
        }
        for (int i = 0; i < array.length(); ++i) {
            JSONObject item = array.getJSONObject(i);
            int tokenId = item.getInt("token_id");
            String token = idToToken.get(tokenId);
            if (token == null) {
                token = "<id:" + tokenId + ">";
            }
            result.add(new FrameTokenProb(
                    tokenId, token, (float) item.getDouble("prob")));
        }
        return result;
    }

    @Override
    public String toString() {
        return tokenId + ":" + token + "=" + String.format("%.6f", prob);
    }
}
