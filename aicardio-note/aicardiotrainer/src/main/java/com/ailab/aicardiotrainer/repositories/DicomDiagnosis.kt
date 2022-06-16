/*
 * Copyright 2020 UET-AILAB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.ailab.aicardiotrainer.repositories

import android.util.Log
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject

class DicomDiagnosis: JSONObject {
    companion object {
        const val TAG = "DicomDiagnosis"
        const val TOOL_NAME = "dicomDiagnosis"
        const val CHAMBER = "chamber"
        const val CHAMBER_IDX = "chamber_idx"
        const val NOT_STANDARD = "not_standard"
        const val POINTS = "points"
        const val LAD = "lad"
        const val RCA = "rca"
        const val LCX = "lcx"
        const val NOTE = "note"

        const val NORMAL_ACTIVITY = -1
        const val ANOMALY_REDUCED_ACTIVITY = 0
        const val ANOMALY_NO_ACTIVITY = 1
        const val ANOMALY_TWISTED_ACTIVITY = 2
        const val ANOMALY_NOT_SYNC_ACTIVITY = 3

        fun getChamberName(index: Int) : String {
            return when (index) {
                0 -> "2C"
                1 -> "3C"
                2 -> "4C"
                3 -> "PTS_L"
                4 -> "PTS_S"
                5 -> "NO"
                else -> "LABEL"
            }
        }

        fun getChamberIdxFromName(name: String) : Int {
            return when (name) {
                "2C" -> 0
                "3C" -> 1
                "4C" -> 2
                "PTS_L" -> 3
                "PTS_S" -> 4
                "NO" -> 5
                else -> -1
            }
        }

        fun getPointType(point: JSONObject): Int {
            try {
                return point.getInt("type")

            } catch (e: JSONException) {
                Log.w(TAG, "getPointType ${point} ${e}")
                return -1
            }
        }
    }

    constructor() : super() {
        put(CHAMBER, getChamberName(-1))
        put(CHAMBER_IDX, -1)
        put(NOT_STANDARD, false)
        put(LAD, false)
        put(RCA, false)
        put(LCX, false)
        put(POINTS, JSONArray())
        put(NOTE, "")
    }
    constructor(s: String) : super(s) {
        if (!this.has(CHAMBER)) put(CHAMBER, getChamberName(-1))
        if (!this.has(CHAMBER_IDX)) put(CHAMBER_IDX, -1)
        if (!this.has(NOT_STANDARD)) put(NOT_STANDARD, false)
        if (!this.has(LAD)) put(LAD, false)
        if (!this.has(RCA)) put(RCA, false)
        if (!this.has(LCX)) put(LCX, false)
        if (!this.has(POINTS)) put(POINTS, JSONArray())
        if (!this.has(NOTE)) put(NOTE, "")
    }

    var points : JSONArray
        get() = getJSONArray(POINTS)
        set(value) { put(POINTS, value) }

    fun clearPoints() {
        points = JSONArray()
    }

    fun addPoint(ix: Float, iy: Float, atype: Int) {
        val o = JSONObject()
        o.put("x", ix)
        o.put("y", iy)
        o.put("type", atype)
        points.put(o)
    }

    fun getPoint(i: Int): JSONObject {
        try {
            return points.getJSONObject(i)
        } catch (e: JSONException) {
            Log.w(TAG, "getPoint ${i} ${e}")
            return JSONObject()
        }
    }

    val nPoints: Int get() = points.length()


    var chamberIdx: Int
        get() {
            return try {
                 getInt(CHAMBER_IDX)
            } catch (e: JSONException) {
                Log.w(TAG, "GET chamberIdx ${e}")
                -1
            }
        }
        set(value) {
            put(CHAMBER_IDX, value)
            put(CHAMBER, getChamberName(value))
        }

    val chamber: String
        get() {
            return try {
                 getString(CHAMBER)
            } catch (e: JSONException) {
                Log.w(TAG, "GET chamber ${e}")
                "LABEL"
            }
        }
//        get() = getString(CHAMBER)

    var lad : Boolean
        get() {
            return try {
                getBoolean(LAD)
            } catch (e: JSONException) {
                Log.w(TAG, "GET lad ${e}")
                false
            }
        }
        set(value) { put(LAD, value) }

    var lcx : Boolean
        get() {
            return  try {
                getBoolean(LCX)
            } catch (e: JSONException) {
                Log.w(TAG, "GET lcx ${e}")
                false
            }
        }
        set(value) { put(LCX, value) }

    var rca : Boolean
        get() {
            return  try {
                getBoolean(RCA)
            } catch (e: JSONException) {
                Log.w(TAG, "GET rca ${e}")
                false
            }
        }
        set(value) { put(RCA, value) }

    var isNotStandardImage : Boolean
        get() {
            return  try {
                getBoolean(NOT_STANDARD)
            } catch (e: JSONException) {
                Log.w(TAG, "GET isNotStandardImage ${e}")
                false
            }
        }
        set(value) { put(NOT_STANDARD, value) }

    var note: String
        get() {
            return  try {
                getString(NOTE)
            } catch (e: JSONException) {
                Log.w(TAG, "GET note ${e}")
                ""
            }
        }
        set(value) { put(NOTE, value) }
}

