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

import aacmvi.AacMviFragment
import android.app.Activity
import android.content.Context
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.fragment.app.viewModels
import com.uetailab.aipacs.R
import com.uetailab.aipacs.home_pacs.HomePacsAPI
import com.uetailab.aipacs.home_pacs.HomePacsActivity
import com.uetailab.aipacs.home_pacs.fragment_intepretation.DicomInterpretation
import kotlinx.android.synthetic.main.fragment_home_view.*
import kotlinx.android.synthetic.main.fragment_home_view_dialog_progress.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch

class HomeViewFragment: AacMviFragment<HomeViewState, HomeViewEffect, HomeViewEvent, HomeViewVM>(),
    HomePacsAPI.ProgressDownloadListener, OnStudyPreviewClicked {

    companion object {

        const val TAG = "HomeViewFragment"

    }

    val studyGVAdapter by lazy {
        HomeViewStudyGVAdapter(context as Context, this, viewModel)
    }

    interface OnHomViewVMPass {
        fun onHomeViewVMPass(homeViewVM: HomeViewVM)
    }

    lateinit var homViewVMPasser: OnHomViewVMPass
    val downloadDataStudyMVIListener = HomeViewDownloadDataMVIListener(this)

    override fun onAttach(context: Context) {
        super.onAttach(context)
        homViewVMPasser = context as OnHomViewVMPass
    }


    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View? {
        super.onCreateView(inflater, container, savedInstanceState)
        val view = inflater.inflate(R.layout.fragment_home_view, container, false)
        return view
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        (activity as HomePacsActivity).getHomeViewVM()?.let {
            viewModel.onGetDataHomeViewFromActivity(it)
            viewModel.viewStates().value?.getListStudyGVItem()?.let { it1 -> studyGVAdapter.submitList(it1) }
            val cases = viewModel.getStudies()
            Log.w(TAG, "onViewCreated Size of cases: ${cases.size}")
            val adapter = ArrayAdapter(context as Context,
                android.R.layout.simple_list_item_1, cases)
            auto_complete_case_index.setAdapter(adapter)
            it.studyID?.let { it1 -> auto_complete_case_index.setText(it1.trimStart('0')) }

            auto_complete_case_index.setOnItemClickListener { parent, view, position, id ->
                Log.w(TAG, "Case id get: ${auto_complete_case_index.text}")
                val studyID = auto_complete_case_index.text.toString()
                if (studyID.length > 0 && studyID.toInt() > 0) {
                    HomeViewFetchDataMVI.process(viewModel, HomeViewEvent.FetchStudy(studyID=studyID.toInt()))
                }
            }


        }


        ib_case_search.setOnClickListener {
            Log.w(TAG, "Case id get: ${auto_complete_case_index.text}")
            val studyID = auto_complete_case_index.text.toString()
            if (studyID.length > 0 && studyID.toInt() > 0) {
//                Log.w(TAG, "Case id get check value: ${studyID.toInt()}")
                HomeViewFetchDataMVI.process(viewModel, HomeViewEvent.FetchStudy(studyID=studyID.toInt()))
            }

        }

        gv_study_preview.apply {
            adapter = studyGVAdapter
        }

    }

    var progressDialog : HomeViewProgressDialog? = null

    fun openProgressDialog() {
        progressDialog = HomeViewProgressDialog(activity as Activity)
        progressDialog!!.setCanceledOnTouchOutside(false)
        progressDialog!!.show()
    }

    fun closeProgressDialog() {
        progressDialog?.let {
            it.dismiss()
            progressDialog = null
        }
    }

    override fun update(bytesRead: Long, contentLength: Long, done: Boolean) {
        Log.w(TAG, "Process to: ${bytesRead} ${contentLength} DONE: ${(100 * bytesRead) / contentLength} %")
        GlobalScope.launch(Dispatchers.Main) {
//            if (done) closeProgressDialog()
            progressDialog?.let {
                it.tv_progress_percentage.text = ((100 * bytesRead) / contentLength).toString()
                it.pb_progress.progress = ((100 * bytesRead) / contentLength).toInt()
            }
        }
    }


    override val viewModel: HomeViewVM by viewModels()


    override fun renderViewState(viewState: HomeViewState) {
        when (viewState.status) {

            HomeViewStatus.Start -> {
                HomeViewFetchDataMVI.process(viewModel, HomeViewEvent.FetchStudies())
            }
            is HomeViewStatus.FetchedErrorData -> {
                closeProgressDialog()
            }
            is HomeViewStatus.FetchedData -> {
                closeProgressDialog()

                homViewVMPasser.onHomeViewVMPass(viewModel)

                when (viewState.status.viewEvent) {

                    is HomeViewEvent.FetchFileMP4 -> {
                        viewState.relativePath?.let {
                            showDicomPreviewDialog(it, viewState.bitmaps)
                        }


                    }
                    is HomeViewEvent.FetchStudy -> {
                        if (viewState.studyID != null && viewState.studyInstanceUID != null)
                            downloadDataStudyMVIListener.process(viewModel, HomeViewEvent.FetchPreviewStudy(studyID=viewState.studyID, studyInstanceUID=viewState.studyInstanceUID))
                    }

                    is HomeViewEvent.FetchPreviewStudy -> {
                        studyGVAdapter.submitList(viewState.getListStudyGVItem())
                    }

                    is HomeViewEvent.FetchStudies -> {
                        val cases = viewModel.getStudies()
                        Log.w(TAG, "Size of cases: ${cases.size}")
                        val adapter = ArrayAdapter(context as Context,
                            android.R.layout.simple_list_item_1, cases)
                        auto_complete_case_index.setAdapter(adapter)

                        auto_complete_case_index.setOnItemClickListener { parent, view, position, id ->
                            Log.w(TAG, "Case id get: ${auto_complete_case_index.text}")
                            val studyID = auto_complete_case_index.text.toString()
                            if (studyID.length > 0 && studyID.toInt() > 0) {
                                HomeViewFetchDataMVI.process(viewModel, HomeViewEvent.FetchStudy(studyID=studyID.toInt()))
                            }
                        }
                    }
                }


            }

            is HomeViewStatus.OnFetchingData -> {
                openProgressDialog()
            }

        }
    }

    fun showDicomPreviewDialog(file: String, bitmaps: List<Bitmap>) {
        val dlg = HomeViewStudyPreviewDialog(activity as HomePacsActivity, file, bitmaps )
        dlg.setCanceledOnTouchOutside(false)
        dlg.show()
    }


    override fun renderViewEffect(viewEffect: HomeViewEffect) {
//        FetchDataHomeMVI.renderViewEffect(viewModel, viewEffect)
        when (viewEffect) {
            is HomeViewEffect.ShowToast -> {
                Toast.makeText(this.context, "${viewEffect.message}", Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onStudyPreviewClicked(item: StudyGVItem) {

//        dataPasser.onViewModelPass(viewModel)
//        TODO("Not yet implemented")
        Log.w(TAG, "${item} ${item.name}")
        val relativePath = viewModel.getRelativePath(item.name)
        if (viewModel.studyID != null && viewModel.studyInstanceUID != null && relativePath != null)
            downloadDataStudyMVIListener.process(viewModel, HomeViewEvent.FetchFileMP4(viewModel.studyID!!, viewModel.studyInstanceUID!!, relativePath ))

    }

    override fun onStudyPreviewLongClicked(item: StudyGVItem): Boolean {
//        TODO("Not yet implemented")
        return true
    }

}