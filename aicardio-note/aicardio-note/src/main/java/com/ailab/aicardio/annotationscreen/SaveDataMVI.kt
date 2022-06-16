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

import android.graphics.Color
import android.util.Log
import androidx.lifecycle.viewModelScope
import com.ailab.aicardio.LCE
import com.ailab.aicardio.repository.AnnotationStateEffectObject
import com.ailab.aicardio.repository.AnnotationRepository
import kotlinx.android.synthetic.main.activity_annotate.*
import kotlinx.coroutines.launch
import org.json.JSONObject

class SaveDataMVI {

    companion object {
        // For Singleton instantiation
        @Volatile
        private var instance: SaveDataMVI? = null

        fun getInstance() =
            instance ?: synchronized(this) {
                instance
                    ?: SaveDataMVI()
                        .also { instance = it }
            }
        const val TAG = "SaveDataUniDirectionMVI"

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
        when (viewEffect) {
            is AnnotationViewEffect.ShowSaveDialog -> annotationActivity.showSaveDialog(viewEffect.file, viewEffect.bitmaps, viewEffect.dicomAnnotation, viewEffect.dicomDiagnosis)
            is AnnotationViewEffect.RenderLoginButton -> renderLoginButton(annotationActivity, viewEffect.phone)
            is AnnotationViewEffect.ShowUserLoginDialog -> annotationActivity.showUserLoginDialog()
        }

    }

    private fun renderViewState(annotationActivity: AnnotationActivity, viewState: AnnotationViewState) {
        when(viewState.status) {
            is AnnotationViewStatus.LoginDone -> renderLoginButton(annotationActivity, viewState.phone)
        }

    }

    private fun renderLoginButton(annotationActivity: AnnotationActivity, phone: String) {
        annotationActivity.bt_login.text = phone
        annotationActivity.bt_login.setBackgroundColor(Color.GREEN)
    }

    fun process(annotationActVM: AnnotationActVM, annotationViewEvent: AnnotationViewEvent) {
        when(annotationViewEvent) {

            is AnnotationViewEvent.SaveDiagnosis -> {
                annotationActVM.viewStates().value?.let {
                    annotationActVM.reduceStateEffectObject(

                        it.copy(status = AnnotationViewStatus.DiagnosisEntered, dicomDiagnosis = annotationViewEvent.dicomDiagnosis),
                        AnnotationViewEffect.RenderDiagnosisTool(dicomDiagnosis = annotationViewEvent.dicomDiagnosis)
                    )
                }

            }

            AnnotationViewEvent.OnUserLogin -> {
                annotationActVM.reduceStateEffectObject(null, AnnotationViewEffect.ShowUserLoginDialog() )
            }

            is AnnotationViewEvent.OnSaveUserLogin -> {

                val phone = annotationViewEvent.user.phone

                annotationActVM.viewStates().value?.let {
                    annotationActVM.reduceStateEffectObject(
                        it.copy(phone=phone, status = AnnotationViewStatus.LoginDone),
                        null)
                }


            }

            AnnotationViewEvent.OnSaveData -> {
                annotationActVM.viewStates().value?.let {
                    annotationActVM.reduceStateEffectObject(
                        null,
                        AnnotationViewEffect.ShowSaveDialog(it.file, it.bitmaps, it.dicomAnnotation, it.dicomDiagnosis)
                    )
                }

            }

            AnnotationViewEvent.OnSaveConfirmed -> {
                annotationActVM.viewStates().value?.let {
                    annotationActVM.reduce(SaveDataToServerReducerAsync(annotationActVM, it, AnnotationViewEvent.OnSaveDataToServer))
                    annotationActVM.reduce(SaveDataToDiskReducerAsync(annotationActVM, it, AnnotationViewEvent.OnSaveDataToDisk(pushNotification=true)))
                }
            }

            is AnnotationViewEvent.OnSaveDataToDisk -> {
                annotationActVM.viewStates().value?.let {
                    annotationActVM.reduce(SaveDataToDiskReducerAsync(annotationActVM, it, AnnotationViewEvent.OnSaveDataToDisk(pushNotification=false)))
                }
            }

            is AnnotationViewEvent.SaveDataToDiskError -> {
                annotationActVM.reduceStateEffectObject(annotationViewEvent.viewState, annotationViewEvent.viewEffect)
            }

            is AnnotationViewEvent.SaveDataToDiskSuccess -> {
                annotationActVM.reduceStateEffectObject(annotationViewEvent.viewState, annotationViewEvent.viewEffect)
            }

            is AnnotationViewEvent.SaveDataToServerError -> {
                annotationActVM.reduceStateEffectObject(annotationViewEvent.viewState, annotationViewEvent.viewEffect)
            }
            is AnnotationViewEvent.SaveDataToServerSuccess -> {
                annotationActVM.reduceStateEffectObject(annotationViewEvent.viewState, annotationViewEvent.viewEffect)

            }
        }
    }

    inner class SaveDataToServerReducerAsync(viewModel: AnnotationActVM, viewState: AnnotationViewState, val viewEvent: AnnotationViewEvent.OnSaveDataToServer)
        : AnnotationActReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): AnnotationStateEffectObject {
            Log.w(TAG, "SaveDataToServerReducerAsync")

            val data = JSONObject()

            data.put(AnnotationActVM.PHONE, viewState.phone)
            data.put(AnnotationActVM.FILE, viewState.file)
            data.put(AnnotationActVM.MANUAL_DIAGNOSIS, viewState.dicomDiagnosis)
            data.put(AnnotationActVM.MANUAL_ANNOTATION, viewState.dicomAnnotation)

            data.put(AnnotationActVM.MACHINE_ANNOTATION, viewState.machineAnnotation)
            data.put(AnnotationActVM.MACHINE_DIAGNOSIS, viewState.machineDiagnosis)

            data.put(AnnotationActVM.DICOM_TAG, viewState.tagsDicom)
            data.put(AnnotationActVM.KEY_VERSION, AnnotationActVM.VERSION_NUMBER)

//            Log.w(TAG, "${data}")

            viewModel.viewModelScope.launch {
                Log.w(TAG, "saving ${viewState.file} dicomAnnotation to server")
                when (val resultLaunch = annotationRepository.postAnnotation(data)) {
                    is LCE.Result -> {
                        Log.w(TAG, "${resultLaunch.error} ${resultLaunch}")

                        if (resultLaunch.error)  {
                            viewModel.viewStates().value?.let {
                                SaveDataMVI.process(viewModel, AnnotationViewEvent.SaveDataToServerError(
                                    it.copy(status = AnnotationViewStatus.SavedToServerError),
                                    AnnotationViewEffect.ShowToast(message = "saving ${it.file} dicomAnnotation to server failed ${resultLaunch.message}")))
                            }

                        } else {
                            viewModel.viewStates().value?.let {
                                SaveDataMVI.process(viewModel, AnnotationViewEvent.SaveDataToServerSuccess(
                                    it.copy(status = AnnotationViewStatus.SavedToServerSuccess),
                                    AnnotationViewEffect.ShowToast(message = "saving ${it.file} dicomAnnotation to server success ${resultLaunch.message}")))
                            }
                        }

                    }
                }
            }
            return AnnotationStateEffectObject(null, AnnotationViewEffect.ShowToast(message = "start saving ${viewState.file} dicomAnnotation to server"))

        }
    }

    inner class SaveDataToDiskReducerAsync(viewModel: AnnotationActVM, viewState: AnnotationViewState, val viewEvent: AnnotationViewEvent.OnSaveDataToDisk)
        : AnnotationActReducer(viewModel, viewState, viewEvent) {


        override fun reduce(): AnnotationStateEffectObject {
            Log.w(TAG, "SaveDataToDiskReducerAsync")

            val data = JSONObject()
            data.put(AnnotationActVM.PHONE, viewState.phone)
            data.put(AnnotationActVM.FILE, viewState.file)

            data.put(AnnotationActVM.MANUAL_DIAGNOSIS, viewState.dicomDiagnosis)
            data.put(AnnotationActVM.MANUAL_ANNOTATION, viewState.dicomAnnotation)

            data.put(AnnotationActVM.MACHINE_DIAGNOSIS, viewState.machineDiagnosis)
            data.put(AnnotationActVM.MACHINE_ANNOTATION, viewState.machineAnnotation)

            data.put(AnnotationActVM.KEY_VERSION, AnnotationActVM.VERSION_NUMBER)


            var result: AnnotationStateEffectObject = AnnotationStateEffectObject(
                viewState.copy(status = AnnotationViewStatus.SavingToDisk), null)

            val fileName = viewState.file + ".json"
            viewModel.viewModelScope.launch {
                when (val readResult = annotationRepository.saveDataToDisk(fileName, data)) {
                    is LCE.Result -> {
                        if (readResult.error) {


                            viewModel.viewStates().value?.let {
                                SaveDataMVI.process(viewModel,
                                    AnnotationViewEvent.SaveDataToDiskError(
                                        it.copy(status = AnnotationViewStatus.SavedToDiskError),
                                        AnnotationViewEffect.ShowToast(message = readResult.message)))

                            }
//

                        } else {
                            viewModel.viewStates().value?.let {
                                SaveDataMVI.process(viewModel,
                                    AnnotationViewEvent.SaveDataToDiskSuccess(
                                        it.copy(status = AnnotationViewStatus.SavedToDiskSuccess),
                                    if (viewEvent.pushNotification)  AnnotationViewEffect.ShowToast(message = readResult.message)
                                    else null))

                            }

                        }
                    }
                }
            }

            return result
        }

    }

}