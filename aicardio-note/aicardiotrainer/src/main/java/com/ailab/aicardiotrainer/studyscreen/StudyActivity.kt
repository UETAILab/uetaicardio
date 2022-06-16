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

import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import androidx.activity.viewModels
import com.ailab.aicardiotrainer.R
import com.ailab.aicardiotrainer.annotationscreen.AnnotationActivity
import com.ailab.aicardiotrainer.api.ProgressListener
import com.ailab.aicardiotrainer.interfaces.OnDicomPreviewClicked
import com.ailab.aicardiotrainer.interpretation.InterpretationActivity
import com.ailab.aicardiotrainer.repositories.StudyItem
import com.ailab.aicardiotrainer.toast
import com.rohitss.aacmvi.AacMviActivity
import kotlinx.android.synthetic.main.activity_study.*
import kotlinx.android.synthetic.main.dialog_progress.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch

class StudyActivity
    :  AacMviActivity<StudyViewState, StudyViewEffect, StudyViewEvent, StudyActVM>(), OnDicomPreviewClicked, ProgressListener {

    companion object {
        const val INTENT_SKILL_NAME = "KEY_SKILL_NAME"
        const val INTENT_ID_CASE = "KEY_ID_CASE"
        const val INTENT_STUDY_NAME = "KEY_STUDY_NAME"

        fun createIntent(context: Context, skillName: String, study: StudyItem): Intent {
            val intent = Intent(context, StudyActivity::class.java)
            intent.putExtra(INTENT_STUDY_NAME, study.name)
            intent.putExtra(INTENT_SKILL_NAME, skillName)
            return intent
        }

        fun createIntent(context: Context, idCase: Int): Intent {
            val intent = Intent(context, StudyActivity::class.java)
            intent.putExtra(INTENT_ID_CASE, idCase)
            return intent
        }


        const val TAG = "StudyActivity"
    }

    var bitmapHeart : Bitmap? = null
    var bitmapPlay : Bitmap? = null
    var bitmapPause : Bitmap? = null

    val dicomGVAdapter by lazy {
        DicomGVAdapter(this, this, viewModel)
    }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_study)

        gv_dicom_preview.apply {
            adapter = dicomGVAdapter
        }
    }

    override val viewModel: StudyActVM by viewModels()

    val StudyLoadDicomMVI = StudyLoadDicomMVI(this)

    override fun renderViewState(viewState: StudyViewState) {
        StudyLoadDicomMVI.renderViewState(this, viewState)
        when (viewState.status) {

            is StudyViewStatus.LoadedStudyInformation -> {

//                Log.w(TAG, "StudyInstanceUID: ${viewModel.studyInstanceUID}")
//                this.toast("StudyInstanceUID: ${viewModel.studyInstanceUID}")
//                Log
                viewState.getStudyInstanceUID()?.let {
                    StudyLoadDicomMVI.process(viewModel, StudyViewEvent.DownloadJPGPreview(studyInstanceUID = it, studyId = viewState.studyId))

                }

            }
            is StudyViewStatus.Start -> {

//                if (intent.hasExtra(INTENT_STUDY_NAME) && intent.hasExtra(INTENT_SKILL_NAME)) {
//                    val studyName = intent.getStringExtra(INTENT_STUDY_NAME)
//                    val skillName = intent.getStringExtra(INTENT_SKILL_NAME)
//                    Log.w(TAG, "Start study $studyName skill $skillName")
//                    studyName?.let {study -> skillName?.let { skill ->
//                        StudyLoadDicomMVI.process(viewModel, StudyViewEvent.DownloadDicomPreview(studyName = study, skillName = skill))
//                    } }
//
//                }

                if (intent.hasExtra(INTENT_ID_CASE)) {
                    val id_case = intent.getIntExtra(INTENT_ID_CASE, 0)
                    id_case?.let {
//                        StudyLoadDicomMVI.process(viewModel, StudyViewEvent.DownloadJPGPreview(studyId = it))
//                        this.toast("Start case id: ${id_case}")
                        if (id_case > 0)
                            StudyLoadDicomMVI.process(viewModel, StudyViewEvent.GetInformationStudy(studyId = it))
                    }
                }

            }
        }
    }

    override fun renderViewEffect(viewEffect: StudyViewEffect) {
        when(viewEffect) {
            is StudyViewEffect.ShowToast -> {
                toast(viewEffect.message)
            }
        }
    }

    override fun onDicomPreviewClicked(item: DicomItem) {
        Log.w(TAG, "onDicomPreviewClicked ${item} ${viewModel.getFileName(item.name)} ${viewModel.studyInstanceUID}")
        viewModel.studyInstanceUID?.let {studyInstanceUID ->
            val relativePath = viewModel.getFileName(item.name)
            relativePath?.let {
                StudyLoadDicomMVI.process(viewModel, StudyViewEvent.DownloadAndExtractMP4File(viewModel.studyId, studyInstanceUID, it ))
            }

        }
//        StudyLoadDicomMVI.process(viewModel, StudyViewEvent.ExtractMP4File("/storage/emulated/0/Download/000008/1.2.840.113663.1500.336282967424147292350824909495184728____IM_0174.mp4"))
//        StudyLoadDicomMVI.
//        val intent = AnnotationActivity.createIntent(this, item)
//        startActivity(intent)
//        showDicomPreviewDialog()

    }

    override fun onDicomPreviewLongClicked(item: DicomItem): Boolean {
        Log.w(TAG, "onDicomPreviewLongClicked ${item}")
//        showDicomPreviewDialog()
        viewModel.studyInstanceUID?.let {study ->
            viewModel.currentFileMP4Path?.let {
                val intent = InterpretationActivity.createIntent(this, study, it)
                startActivity(intent)
            }

        }
        return true
    }

    fun showDicomPreviewDialog(file: String, bitmaps: List<Bitmap>) {
        val dlg = DicomPreviewDialog(this, file, bitmaps )
        dlg.setCanceledOnTouchOutside(false)
        dlg.show()
    }

    override fun update(bytesRead: Long, contentLength: Long, done: Boolean) {
        if (contentLength == -1L) return
        Log.w(AnnotationActivity.TAG, "Process to: ${bytesRead} ${contentLength} DONE: ${(100 * bytesRead) / contentLength} % done: ${done}")
        GlobalScope.launch(Dispatchers.Main) {
            if (done) closeDownloadProgressDialog()
            downloadProgressDialog?.let {
                it.tv_progress_percentage.text = ((100 * bytesRead) / contentLength).toString()
                it.pb_progress.progress = ((100 * bytesRead) / contentLength).toInt()
            }
        }
    }

    var downloadProgressDialog : DownloadProgressDialog? = null

    fun openDownloadProgressDialog() {
        Log.w(TAG, "openDownloadProgressDialog")
        downloadProgressDialog = DownloadProgressDialog(this)
        downloadProgressDialog!!.setCanceledOnTouchOutside(false)
        downloadProgressDialog!!.show()
    }

    fun closeDownloadProgressDialog() {
        downloadProgressDialog?.let {
            it.dismiss()
            downloadProgressDialog = null
        }
    }


}
