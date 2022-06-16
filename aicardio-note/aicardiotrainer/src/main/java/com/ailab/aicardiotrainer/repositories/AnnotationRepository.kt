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
import com.ailab.aicardiotrainer.LCE
import com.ailab.aicardiotrainer.api.AnnotationApi
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.*

class AnnotationRepository {
    companion object {

        // For Singleton instantiation
        @Volatile
        private var instance: AnnotationRepository? = null

        fun getInstance() =
            instance ?: synchronized(this) {
                instance
                    ?: AnnotationRepository()
                        .also { instance = it }
            }
    }


    suspend fun postAnnotation(data: JSONObject) : LCE<JSONObject> = withContext(Dispatchers.IO) {

//        Log.w("response", "${AnnotationApi.create().postAnnotation(data)}")

        var response = JSONObject()

        val annotationApiResult = try {
            val bodyRequest: RequestBody =
                data.toString().toRequestBody("application/json".toMediaTypeOrNull())
//            val b = RequestBody.create(MediaType.parse("application/json"), data.toString());
            response = JSONObject(AnnotationApi.create().postAnnotation(bodyRequest).string())
//            val call = AnnotationApi.create().postAnnotation(bodyRequest).
//
//            Log.w("annotationApiResult value", "${w.get("data")}")

        } catch (e: Exception) {
            Log.w("errorPostAnnotation", "$e")
//            Log.w("postAnnotation", "$data $e")
            return@withContext LCE.Result(data = JSONObject(), error = true, message = "annotationApiResult error")
        }
//        Log.w("dataGETAuto", "${annotationApiResult.component1()}")

        return@withContext if (true) {
            LCE.Result(data = response, error = false, message = "no error")
        } else {
            LCE.Result(data = JSONObject(), error = true, message = "status not ok")
        }
    }



    suspend fun saveDataToDisk(fileName: String, data: JSONObject): LCE<String> = withContext(Dispatchers.IO) {
        try {
            val file = File(fileName)
            val out = OutputStreamWriter(FileOutputStream(file) as OutputStream)
            out.write(data.toString())
            out.close()
            return@withContext LCE.Result(error = false, message = "Save to Disk Result", data = "None")
        } catch (e: IOException) {
//            e.printStackTrace()
//            return@withContext LCE.Error(message = "Save to Disk Falied")
            return@withContext LCE.Result(error = true, message = "Save to Disk Failed", data = "None")


        }
    }

    suspend fun getAnnotationFromFile(fileName: String) : LCE<JSONObject> = withContext(Dispatchers.IO){

       try {
            val inp = InputStreamReader(FileInputStream(fileName))
            val str = inp.readText()
            inp.close()
           return@withContext LCE.Result(error = false, message = "Get DATA from Disk Succeed", data = JSONObject(str))

        } catch (e: Exception) {
           return@withContext LCE.Result(error = true, message = "Get DATA from Disk Failed", data = JSONObject())
       }
    }


}