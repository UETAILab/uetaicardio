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

import com.example.listfolder.repository.NewsItem

data class ListFolderState(val fetchStatus: FetchStatus, val newsList: List<NewsItem>)

sealed class ListFolderEffect {
    data class ShowSnackbar(val message: String) : ListFolderEffect()
    data class ShowToast(val message: String) : ListFolderEffect()
}

sealed class ListFolderEvent {
    data class NewsItemClicked(val newsItem: NewsItem) : ListFolderEvent()
    object FabClicked : ListFolderEvent()
    object OnSwipeRefresh : ListFolderEvent()
    object FetchNews : ListFolderEvent()
}

sealed class FetchStatus {
    object Fetching : FetchStatus()
    object Fetched : FetchStatus()
    object NotFetched : FetchStatus()
}