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

package com.ailab.aicardiotrainer.studyscreen

import android.app.Application
import android.util.Log
import com.rohitss.aacmvi.AacMviViewModel
import org.json.JSONObject
import java.io.File.pathSeparator


class StudyActVM(application: Application) : AacMviViewModel<StudyViewState, StudyViewEffect, StudyViewEvent>(application) {
    companion object {
        const val TAG = "StudyActVM"
    }
    init {
        viewState = StudyViewState(
            status = StudyViewStatus.Start,
            skillName = "NO-SKILL",
            studyName = "NO-NAME",
            studyId = "NO-STUDY-ID",
            studyPaths = emptyList(),
            studyInformation = JSONObject(),
            currentFileMP4Path = null,
            currentBitmap = emptyList()
        )
    }
    val studyId get() = viewState.studyId
    val currentFileMP4Path get() = viewState.currentFileMP4Path
    val currentBitmap get() = viewState.currentBitmap
    val studyName get() = viewState.studyName
    val studyInformationStudy get() = viewState.studyInformation
    val listFileDicom: JSONObject get() = if (studyInformationStudy.has("ListFileDicom")) studyInformationStudy.getJSONObject("ListFileDicom") else JSONObject()
    fun getDicomViewAndNumberOfFrame(fileName: String): Pair<String, String> {
//        Log.w(TAG, "fileName: ${fileName} getDicomViewAndNumberOfFrame: ${newFileName}")
        listFileDicom.keys().forEach {
//            Log.w(TAG, "Key: ${it}")
            if (it.contains(fileName)) {
                try {
                    val obj = listFileDicom.getJSONObject(it)
                    return Pair(obj.getJSONObject("DicomView").getJSONObject("DataView").getString("View"), obj.getInt("NumberOfFrames").toString() + " --- " + obj.getInt("InstanceNumber").toString()  )
                } catch (e: Exception) {
                    Log.w(TAG, "getDicomViewAndNumberOfFrame: ${e}")
                    return Pair("No-DicomView", "1")
                }
            }
        }
        return Pair("No-DicomView", "1")

    }

    fun getFileName(itemName: String): String? {
        val shortNameArr = itemName.split("____")
        val fileName = shortNameArr[shortNameArr.size - 1]
        listFileDicom.keys().forEach {
//            Log.w(TAG, "Key: ${it}")
            if (it.contains(fileName)) {
                try {
                    val obj = listFileDicom.getJSONObject(it)
                    return obj.getString("RelativePath")
                } catch (e: Exception) {
                    Log.w(TAG, "getDicomViewAndNumberOfFrame: ${e}")
                    return null
                }
            }
        }
        return null

    }
    val studyInstanceUID get() = viewState.getStudyInstanceUID()

    fun reduce(reducer: StudyActReducer) {
        val result = reducer.reduce()
        reduce(result)
    }

    fun reduce(result: StudyStateEffectObject) {
        result.viewState?.let { viewState = it }
        result.viewEffect?.let { viewEffect = it }
    }

}