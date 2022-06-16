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

package com.ailab.aicardio.annotationscreen

import android.util.Log
import androidx.lifecycle.viewModelScope
import com.ailab.aicardio.LCE
import com.ailab.aicardio.R
import com.ailab.aicardio.repository.AnnotationStateEffectObject
import com.ailab.aicardio.repository.AnnotationRepository
import com.ailab.aicardio.repository.DicomAnnotation
import com.ailab.aicardio.repository.DicomDiagnosis
import kotlinx.android.synthetic.main.activity_annotate.*
import kotlinx.coroutines.launch
import org.json.JSONException
import org.json.JSONObject

class AutoAnalysisMVI {

    companion object {
        // For Singleton instantiation
        @Volatile
        private var instance: AutoAnalysisMVI? = null

        fun getInstance() =
            instance ?: synchronized(this) {
                instance
                    ?: AutoAnalysisMVI()
                        .also { instance = it }
            }

        const val TAG = "AutoAnalysisUniDirectionMVI"
        const val KEY_SIUID = "siuid"
        const val KEY_GET = "get"
        const val KEY_SopIUID = "sopiuid"
        const val KEY_PATH_FILE = "path_file"
        const val KEY_DATA_FROM_SERVER = "data"

        fun process(annotationActVM: AnnotationActVM, annotationViewEvent: AnnotationViewEvent) {
            getInstance().process(annotationActVM, annotationViewEvent)
        }

        fun renderViewState(annotationActivity: AnnotationActivity, viewState: AnnotationViewState) {
            getInstance().renderViewState(annotationActivity, viewState)
        }

        fun renderViewEffect(annotationActivity: AnnotationActivity, viewEffect: AnnotationViewEffect) {
            getInstance().renderViewEffect(annotationActivity, viewEffect)
        }
    }
    private val annotationRepository = AnnotationRepository.getInstance()


    private fun renderViewEffect(annotationActivity: AnnotationActivity, viewEffect: AnnotationViewEffect) {

    }

    private fun renderViewState(annotationActivity: AnnotationActivity, viewState: AnnotationViewState) {
        when (viewState.status) {
//            is AnnotationViewStatus.Test -> {
////                val newText = "-- ${viewState.annotateViewStatus.message} --"
////                annotationActivity.bt_test.text = newText
////            }
////            is AnnotationViewStatus.TestAsync -> {
////                val newText = "++ ${viewState.annotateViewStatus.message} ++"
////                annotationActivity.bt_test.text = newText
////            }
//            is AnnotationViewStatus.
            AnnotationViewStatus.AutoAnnalysisFetching -> {
                annotationActivity.bt_auto_analysis.background = annotationActivity.resources.getDrawable(R.drawable.custom_button_enabled_color, null)
                Log.w(TAG, "AnnotationViewStatus.AutoAnnalysisFetching")

            }
            AnnotationViewStatus.AutoAnnalysisFetched -> {
                annotationActivity.bt_auto_analysis.background = annotationActivity.resources.getDrawable(R.drawable.custom_button_disabled_color, null)
                Log.w(TAG, "AnnotationViewStatus.AutoAnnalysisFetched")
                annotationActivity.iv_draw_canvas.invalidate()
            }
        }
    }

    fun process(annotationActVM: AnnotationActVM, annotationViewEvent: AnnotationViewEvent) {
        when (annotationViewEvent) {
            is AnnotationViewEvent.AutoAnalysis -> {
                annotationActVM.viewStates().value?.let {
                    annotationActVM.reduce(Reducer(annotationActVM, it, annotationViewEvent))
                }
            }
//            is AnnotationViewEvent.ToggleAutoDraw -> {
//                annotationActVM.setAutoDraw(annotationViewEvent.isAutoDraw)
//            }

        }
    }

    inner class Reducer(viewModel: AnnotationActVM, viewState: AnnotationViewState, val viewEvent: AnnotationViewEvent.AutoAnalysis)
        : AnnotationActReducer(viewModel, viewState, viewEvent) {

//        suspend fun asyncRun() : Int = withContext(Dispatchers.IO) {
//            repeat(100000000) {
//                100*100
//            }
//            return@withContext 1234
//        }

        override fun reduce(): AnnotationStateEffectObject {


            val data = POSTAutoAnalysisToServer(viewState)

            viewModel.viewModelScope.launch {
                Log.w(AnnotationActVM.TAG, "get auto analysis ${viewState.file} dicomAnnotation from server")
                when (val resultLaunch = annotationRepository.postAnnotation(data)) {
                    is LCE.Result -> {
//                    Log.w(TAG, "${resultLaunch.error} ${resultLaunch}")

                        if (resultLaunch.error)  {

                            viewModel.viewStates().value?.let {
                                viewModel.reduce(ReducerAsync(viewModel, it, AnnotationViewEvent.AutoAnalysisAsyncError))
                            }


                        }
                        else {
                            try {
                                val resultResponse = resultLaunch.data.getJSONObject(KEY_DATA_FROM_SERVER)

                                val dicomAnnotation = DicomAnnotation(resultResponse.getJSONArray(AnnotationActVM.MANUAL_ANNOTATION))

                                val dicomDiagnosis = DicomDiagnosis(resultResponse.getJSONObject(AnnotationActVM.MANUAL_DIAGNOSIS).toString())

                                viewModel.viewStates().value?.let {
                                    viewModel.reduce(ReducerAsync(viewModel, it, AnnotationViewEvent.AutoAnalysisAsyncSuccess(dicomAnnotation, dicomDiagnosis)))
                                }

                            } catch (e: JSONException) {

                                Log.w(TAG, "ReducerAsync ${e}")

                                viewModel.viewStates().value?.let {
                                    viewModel.reduce(ReducerAsync(viewModel, it, AnnotationViewEvent.AutoAnalysisAsyncError))
                                }
                            }

                        }

                    }
                }
            }

            return AnnotationStateEffectObject(viewState.copy(status = AnnotationViewStatus.AutoAnnalysisFetching), null)
        }

    }

    inner class ReducerAsync(viewModel: AnnotationActVM, viewState: AnnotationViewState, val viewEvent: AnnotationViewEvent) : AnnotationActReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): AnnotationStateEffectObject {

            when (viewEvent) {
                AnnotationViewEvent.AutoAnalysisAsyncError -> {
                    return AnnotationStateEffectObject(
                        viewState.copy(status = AnnotationViewStatus.AutoAnnalysisFetched, machineAnnotation = DicomAnnotation.getNewAnnotation(viewModel.numFrame), machineDiagnosis = DicomDiagnosis()),
                        AnnotationViewEffect.ShowToast("Auto Analysis error Loading")
                    )
                }
                is AnnotationViewEvent.AutoAnalysisAsyncSuccess -> {
                    return AnnotationStateEffectObject(
                        viewState.copy(machineDiagnosis = viewEvent.dicomDiagnosis, machineAnnotation = viewEvent.dicomAnnotation, status = AnnotationViewStatus.AutoAnnalysisFetched),
                        AnnotationViewEffect.ShowToast("Auto Analysis Done Loading")
                    )
                }
            }
            return AnnotationStateEffectObject()
        }

    }

    private fun POSTAutoAnalysisToServer(viewState: AnnotationViewState): JSONObject {
        val data = JSONObject()
        data.put(KEY_PATH_FILE, viewState.file)
        data.put(KEY_SIUID, viewState.sIUID)
        data.put(KEY_SopIUID, viewState.sopIUID)
        data.put(KEY_GET, KEY_GET)
        return data
    }
}