package com.rohitss.mvr.annotatescreen

import com.rohitss.mvr.repository.NewsItem


data class AnnotateViewState(val fetchStatus: FetchStatus, val newsList: List<NewsItem>)

sealed class AnnotateViewEffect {
    data class ShowSnackbar(val message: String) : AnnotateViewEffect()
    data class ShowToast(val message: String) : AnnotateViewEffect()
}

sealed class AnnotateViewEvent {
    data class NewsItemClicked(val newsItem: NewsItem) : AnnotateViewEvent()
    object FabClicked : AnnotateViewEvent()
    object OnSwipeRefresh : AnnotateViewEvent()
    object FetchNews : AnnotateViewEvent()
}

sealed class FetchStatus {
    object Fetching : FetchStatus()
    object Fetched : FetchStatus()
    object NotFetched : FetchStatus()
}