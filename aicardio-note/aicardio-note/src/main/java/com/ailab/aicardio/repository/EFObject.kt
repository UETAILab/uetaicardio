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

package com.ailab.aicardio.repository

import android.util.Log
import org.json.JSONException
import org.json.JSONObject
//
data class EFObject(
    val indexFrame: Int = -1,
    val indexESV: Int = -1,
    val indexEDV: Int = -1,
    val volumeESV: Float = 0F,
    val volumeEDV: Float = 0F,
    val efValue: Float = -1.0F) {
    companion object {
        const val TAG = "EFObject"
        const val KEY_indexFrame = "indexFrame"
        const val KEY_indexESV = "indexESV"
        const val KEY_indexEDV = "indexEDV"
        const val KEY_volumeESV = "volumeESV"
        const val KEY_volumeEDV = "volumeEDV"
        const val KEY_efManual = "ef_manual"

        fun convertToEFObject(o: JSONObject): EFObject {

            val indexFrame = try {
                o.getInt(KEY_indexFrame)
            } catch (e: JSONException) {
                Log.w(TAG, "convertToEFObject indexFrame ${e}")
                -1
            }
            val indexESV = try {
                o.getInt(KEY_indexESV)
            } catch (e: JSONException) {
                Log.w(TAG, "convertToEFObject indexESV ${e}")                -1
            }
            val indexEDV =  try {
                o.getInt(KEY_indexEDV)
            } catch (e: JSONException) {
                Log.w(TAG, "convertToEFObject indexEDV ${e}")
                -1
            }
            val volumeESV = try {
                o.getDouble(KEY_volumeESV).toFloat()
            } catch(e: JSONException) {
                Log.w(TAG, "convertToEFObject volumeESV ${e}")
                0F
            }
            val volumeEDV = try {
                o.getDouble(KEY_volumeEDV).toFloat()
            } catch(e: JSONException) {
                Log.w(TAG, "convertToEFObject volumeEDV ${e}")
                0F
            }
            val efValue = try {
                o.getDouble(KEY_efManual).toFloat()
            } catch(e: JSONException) {
                Log.w(TAG, "convertToEFObject efValue ${e}")
                0F
            }
            return EFObject(indexFrame = indexFrame, indexESV = indexESV, indexEDV = indexEDV, volumeEDV = volumeEDV, volumeESV = volumeESV, efValue = efValue )

        }
    }
//    override fun toString(): String {
//        val o = JSONObject()
//        o.put(KEY_indexFrame, indexFrame)
//        o.put(KEY_indexESV, indexESV)
//        o.put(KEY_indexEDV, indexEDV)
//        o.put(KEY_volumeESV, volumeESV)
//        o.put(KEY_volumeEDV, volumeEDV)
//        o.put(KEY_efManual, efValue)
//        return o.toString()
//    }
}


//}

//class EFObject: JSONObject {
//    constructor() : super()
//    constructor(arr: JSONObject): super(arr.toString())
//
//    override fun put(name: String, value: Int): JSONObject {
//        return super.put(name, value)
//    }
//}