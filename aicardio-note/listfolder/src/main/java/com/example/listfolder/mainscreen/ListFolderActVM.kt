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

import android.app.Application
import androidx.lifecycle.viewModelScope
import com.example.listfolder.repository.NewsItem
import com.example.listfolder.repository.Utils
import com.rohitss.aacmvi.AacMviViewModel
import kotlinx.coroutines.launch
import java.util.logging.Logger

class ListFolderActVM(application: Application) : AacMviViewModel<ListFolderState, ListFolderEffect, ListFolderEvent>(application) {

    init {
        viewState = ListFolderState(fetchStatus = FetchStatus.NotFetched, newsList = emptyList())
    }
    companion object {
        const val TAG = "ListFolderActVM"
    }

    override fun process(viewEvent: ListFolderEvent) {
        super.process(viewEvent)
        when (viewEvent) {
//
//            is AnnotateViewEvent.NewsItemClicked -> newsItemClicked(viewEvent.newsItem)
//            AnnotateViewEvent.FabClicked -> fabClicked()
//            AnnotateViewEvent.OnSwipeRefresh -> fetchNews()

            ListFolderEvent.FetchNews -> populateFolderList()

            ListFolderEvent.OnSwipeRefresh -> populateFolderList()

        }
    }



    private fun populateFolderList() {
        viewState = viewState.copy(fetchStatus = FetchStatus.Fetching)
        viewModelScope.launch {

            val arrayFolder = arrayListOf<String>()

            Utils.walk(null, arrayFolder, false )

            var dataItems = arrayListOf<NewsItem>()



            arrayFolder.forEach {
                dataItems.add(NewsItem(path = it.toString(), modifiedTime = it.toString(), image = it.toString(), isWorkedOn = false))
            }



            viewState = viewState.copy(fetchStatus = FetchStatus.Fetched, newsList = dataItems)

//            when (val result = repository.getSetLatestNewsFromServer()) {
//                is LCE.Error -> {
//                    viewState = viewState.copy(fetchStatus = FetchStatus.Fetched)
//                    viewEffect = AnnotateViewEffect.ShowToast(message = result.message)
//                }
//                is LCE.Success -> {
//                    viewState =
//                        viewState.copy(fetchStatus = FetchStatus.Fetched, newsList = result.data)
//                }
//            }
        }
    }

}