package com.dan.recture.persistance;

import java.util.List;

import com.dan.recture.domain.TextAnalysisModel;

public interface TextAnalysisDAO {
	//VO를 가지고 실제로 접근하는 애
	//물리적
	//mapper까지 가게해줌
	
	//public List<TextAnalysisModel> list(TextAnalysisModel result)throws Exception;
	public List<TextAnalysisModel> Result() throws Exception;
}
