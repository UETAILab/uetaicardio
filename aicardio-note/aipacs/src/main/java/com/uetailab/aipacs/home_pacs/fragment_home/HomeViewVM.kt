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

package com.uetailab.aipacs.home_pacs.fragment_home
import aacmvi.AacMviViewModel
import android.app.Application
import android.util.Log
import org.json.JSONObject

class HomeViewVM(application: Application) : AacMviViewModel<HomeViewState, HomeViewEffect, HomeViewEvent>(application) {
    companion object {
        const val TAG = "HomeViewVM"
    }
    var cntClicked = 0

    init {
        viewState = HomeViewState(
            status = HomeViewStatus.Start,
            message = "Start-HomeView-Activity",
            studies = emptyList(),
            studyMetaData = JSONObject(),
            studyPreview = emptyList(),
            bitmaps = emptyList(),
            studyInterpretation = JSONObject()
        )
    }
    val studyID get() = viewState.studyID

    val studyInterpretation get() = viewState.studyInterpretation
    val studyInstanceUID get() = viewState.studyInstanceUID
    val studyPreview get() = viewState.studyPreview
    val studyMetaData get() = viewState.studyMetaData
    val listFileDicom: JSONObject get() = if (studyMetaData.has("ListFileDicom")) studyMetaData.getJSONObject("ListFileDicom") else JSONObject()

    fun getMessage(): String {
        return viewState.message
    }
    fun addCountToMessage() {
        cntClicked += 1
        viewState = viewState.copy(message = "${cntClicked}" )
    }

    fun getStudies(): List<Int> {
        return viewState.studies
    }

    fun reduce(reducer: HomeViewReducer) {
        val result = reducer.reduce()
        reduce(result)
    }


    fun reduce(result: HomeViewObject) {
        result.viewState?.let { viewState = it }
        result.viewEffect?.let { viewEffect = it }
    }


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

    fun getRelativePath(itemName: String): String? {
        val shortNameArr = itemName.split("____")
        val fileName = shortNameArr[shortNameArr.size - 1]
        listFileDicom.keys().forEach {
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

    fun onGetDataHomeViewFromActivity(homeViewVM: HomeViewVM) {
        reduce(HomeViewObject(viewState = homeViewVM.viewState, viewEffect = HomeViewEffect.ShowToast("Get Data From Main Activity")))
    }

    fun getHasAnnotation(relativePath: String): Boolean{
        return studyInterpretation.has(relativePath)
    }

}