package com.rohitss.mvr.annotatescreen

import android.app.Application
import android.util.Log
import androidx.lifecycle.viewModelScope
import com.rohitss.aacmvi.AacMviViewModel
import com.rohitss.mvr.LCE
import com.rohitss.mvr.repository.NewsItem
import com.rohitss.mvr.repository.NewsRepository
import kotlinx.coroutines.launch

class MainActVM(application: Application) :
    AacMviViewModel<AnnotateViewState, AnnotateViewEffect, AnnotateViewEvent>(application) {
    private var count: Int = 0
    private val repository: NewsRepository = NewsRepository.getInstance()

    companion object {
        val TAG: String = "MainActVM"
    }

    init {
        viewState = AnnotateViewState(fetchStatus = FetchStatus.NotFetched, newsList = emptyList())
    }

    override fun process(viewEvent: AnnotateViewEvent) {
        super.process(viewEvent)
        when (viewEvent) {
            is AnnotateViewEvent.NewsItemClicked -> newsItemClicked(viewEvent.newsItem)
            AnnotateViewEvent.FabClicked -> fabClicked()
            AnnotateViewEvent.OnSwipeRefresh -> fetchNews()
            AnnotateViewEvent.FetchNews -> fetchNews()
        }
    }

    private fun newsItemClicked(newsItem: NewsItem) {
        viewEffect = AnnotateViewEffect.ShowSnackbar(newsItem.title)
    }

    private fun fabClicked() {
        count++
        viewEffect = AnnotateViewEffect.ShowToast(message = "Fab clicked count $count")
    }

    private fun fetchNews() {
        viewState = viewState.copy(fetchStatus = FetchStatus.Fetching)
        Log.w(TAG, "start fetching")
        viewModelScope.launch {
            when (val result = repository.getSetLatestNewsFromServer()) {
                is LCE.Error -> {
                    viewState = viewState.copy(fetchStatus = FetchStatus.Fetched)
                    viewEffect = AnnotateViewEffect.ShowToast(message = result.message)
                }
                is LCE.Success -> {
                    Log.w(TAG, "fetching success")
                    viewState =
                        viewState.copy(fetchStatus = FetchStatus.Fetched, newsList = result.data)
                }
            }
        }
    }
}