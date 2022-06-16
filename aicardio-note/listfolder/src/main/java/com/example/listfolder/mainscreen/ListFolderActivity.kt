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

package com.example.listfolder.annotatescreen

import android.Manifest
import android.os.Bundle
import androidx.activity.viewModels
import com.example.listfolder.R
import com.example.listfolder.repository.Utils
import com.rohitss.aacmvi.AacMviActivity
import kotlinx.android.synthetic.main.activity_list_folder.*
import java.util.logging.Logger

class ListFolderActivity : AacMviActivity<ListFolderState, ListFolderEffect, ListFolderEvent, ListFolderActVM>() {

    companion object {
        const val TAG = "ListFolderActivity"
    }
    override val viewModel: ListFolderActVM by viewModels()

    private val ListFolderRvAdapter by lazy {
        ListFolderRvAdapter {
//            viewModel.process(FolderEvent.NewsItemClicked(it.tag as NewsItem))

        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {

        super.onCreate(savedInstanceState)

        Utils.checkAndRequestPermissions(
            this,
            arrayOf(
                Manifest.permission.READ_EXTERNAL_STORAGE,
                Manifest.permission.WRITE_EXTERNAL_STORAGE,
                Manifest.permission.INTERNET
            )
        )

        setContentView(R.layout.activity_list_folder)

//        Logger.getLogger(TAG).warning("onCreate")
//
        rvNewsHome.adapter = ListFolderRvAdapter

        srlNewsHome.setOnRefreshListener {
            viewModel.process(ListFolderEvent.OnSwipeRefresh)
        }

    }





//    override val viewModel: ListFolderActVM
//        get() = TODO("Not yet implemented")

    override fun renderViewState(viewState: ListFolderState) {
//        TODO("Not yet implemented")
        Logger.getLogger(TAG).warning("renderViewState")
        when (viewState.fetchStatus) {
            is FetchStatus.Fetched -> {
                srlNewsHome.isRefreshing = false
            }
            is FetchStatus.NotFetched -> {
                viewModel.process(ListFolderEvent.FetchNews)
                srlNewsHome.isRefreshing = false
            }
            is FetchStatus.Fetching -> {
                srlNewsHome.isRefreshing = true
            }
        }
        ListFolderRvAdapter.submitList(viewState.newsList)

    }

    override fun renderViewEffect(viewEffect: ListFolderEffect) {
        TODO("Not yet implemented")
    }

//    fun g
}
