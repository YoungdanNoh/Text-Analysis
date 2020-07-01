package com.dan.recture.service;

import java.util.List;

import com.dan.recture.domain.AirbnbResultVO;


public interface AirbnbResultService {
	public List<AirbnbResultVO> selectResult() throws Exception;
}
