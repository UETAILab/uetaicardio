/*
 * Copyright 2021 UET-AILAB
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

package com.uetailab.aipacs.home_pacs.fragment_intepretation

import android.content.res.Resources
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.InputStreamReader

const val TAG = "InterpretationViewUtils"

fun getFileJSONFromResourcesJSONArray(resources: Resources, idFile: Int) : JSONArray {
    try {
        val inp = InputStreamReader(resources.openRawResource(idFile))

        val str = inp.readText()
        inp.close()
        val result = JSONArray(str)
        return result
//        return@withContext LCE.Result(error = false, message = "Get DATA from Disk Succeed", data = JSONObject(str))

    } catch (e: Exception) {
        Log.w(TAG, "Read getFileJSONFromResources error")
//        return@withContext LCE.Result(error = true, message = "Get DATA from Disk Failed", data = JSONObject())
    }
    return JSONArray()
}

fun getFileJSONFromResourcesJSONObject(resources: Resources, idFile: Int) : JSONObject {
    try {
        val inp = InputStreamReader(resources.openRawResource(idFile))

        val str = inp.readText()
        inp.close()
        val result = JSONObject(str)
        return result
//        return@withContext LCE.Result(error = false, message = "Get DATA from Disk Succeed", data = JSONObject(str))

    } catch (e: Exception) {
        Log.w(TAG, "Read getFileJSONFromResources error")
//        return@withContext LCE.Result(error = true, message = "Get DATA from Disk Failed", data = JSONObject())
    }
    return JSONObject()
}