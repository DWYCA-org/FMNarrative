#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <iostream>
#include <string>
#include <vector>
#include <regex>
#include <map>
#include <algorithm>
#include <sstream>
#include <set>
#include <iterator>

struct MatchStatistic {
    std::string statName;
    std::string homeValue;
    std::string awayValue;
    std::string homeTeam;
    std::string awayTeam;
};

class OCRReader {
private:
    tesseract::TessBaseAPI* ocr;
    std::map<std::string, std::string> statMappings = {
        {"shots", "shots"},
        {"on target", "on target"}, 
        {"xg", "xg"},
        {"off target", "off target"},
        {"clear cut chances", "clear cut chances"},
        {"long shots", "long shots"},
        {"possession", "possession"},
        {"corners", "corners"},
        {"fouls", "fouls"},
        {"offsides", "offsides"},
        {"offside", "offsides"},
        {"passes completed", "passes completed"},
        {"crosses completed", "crosses completed"},
        {"tackles won", "tackles won"},
        {"headers won", "headers won"},
        {"yellow cards", "yellow cards"},
        {"yellow", "yellow cards"},
        {"red cards", "red cards"},
        {"red", "red cards"},
        {"average rating", "average rating"},
        {"progressive passes", "progressive passes"},
        {"high intensity sprints", "high intensity sprints"}
    };

public:
    OCRReader() {
        ocr = new tesseract::TessBaseAPI();
        if (ocr->Init(NULL, "eng")) {
            std::cerr << "Could not initialize tesseract." << std::endl;
            exit(1);
        }
        ocr->SetPageSegMode(tesseract::PSM_AUTO);
    }

    ~OCRReader() {
        if (ocr) {
            ocr->End();
            delete ocr;
        }
    }

    cv::Mat preprocessImage(const cv::Mat& image) {
        cv::Mat gray, processed;

        if (image.channels() > 1) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image.clone();
        }

        // Simple, effective preprocessing
        cv::GaussianBlur(gray, gray, cv::Size(1, 1), 0);
        cv::threshold(gray, processed, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
        cv::resize(processed, processed, cv::Size(), 2.5, 2.5, cv::INTER_CUBIC);

        return processed;
    }

    cv::Mat preprocessImageForTeamNames(const cv::Mat& image) {
        cv::Mat gray, processed;

        if (image.channels() > 1) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image.clone();
        }

        int headerHeight = std::min(150, gray.rows / 3);
        cv::Rect headerRegion(0, 0, gray.cols, headerHeight);
        cv::Mat header = gray(headerRegion);
        
        cv::GaussianBlur(header, header, cv::Size(1, 1), 0);
        cv::adaptiveThreshold(header, header, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, 
                             cv::THRESH_BINARY, 15, 10);
        
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 1));
        cv::morphologyEx(header, header, cv::MORPH_CLOSE, kernel);
        cv::resize(header, header, cv::Size(), 4.0, 4.0, cv::INTER_CUBIC);
        
        return header;
    }

    std::string performOCR(const cv::Mat& image) {
        ocr->SetImage(image.data, image.cols, image.rows, 1, image.cols);
        char* outText = ocr->GetUTF8Text();
        std::string result(outText);
        delete[] outText;
        return result;
    }

    std::vector<std::string> splitLines(const std::string& text) {
        std::vector<std::string> lines;
        std::istringstream stream(text);
        std::string line;

        while (std::getline(stream, line)) {
            line = std::regex_replace(line, std::regex("^\\s+|\\s+$"), "");
            if (!line.empty()) {
                lines.push_back(line);
            }
        }
        return lines;
    }

    std::pair<std::string, std::string> extractTeamNames(const std::vector<std::string>& lines) {
        std::string homeTeam = "", awayTeam = "";
        
        for (size_t i = 0; i < std::min((size_t)10, lines.size()); ++i) {
            std::string line = lines[i];
            std::string upperLine = line;
            std::transform(upperLine.begin(), upperLine.end(), upperLine.begin(), ::toupper);
            
            // Look for two team names in the same line
            std::regex twoTeamsPattern("\\b([A-Z]{4,15})\\b.*\\b([A-Z]{4,15})\\b");
            std::smatch teamMatches;
            if (std::regex_search(upperLine, teamMatches, twoTeamsPattern)) {
                std::string team1 = teamMatches[1].str();
                std::string team2 = teamMatches[2].str();
                
                std::set<std::string> headerWords = {"MATCH", "STATS", "STATISTIC", "STATISTICS"};
                if (headerWords.find(team1) == headerWords.end() && 
                    headerWords.find(team2) == headerWords.end() &&
                    team1 != team2) {
                    homeTeam = team1;
                    awayTeam = team2;
                    break;
                }
            }
        }
        
        // Convert to proper case
        if (!homeTeam.empty()) {
            std::transform(homeTeam.begin(), homeTeam.end(), homeTeam.begin(), ::tolower);
            homeTeam[0] = std::toupper(homeTeam[0]);
        }
        
        if (!awayTeam.empty()) {
            std::transform(awayTeam.begin(), awayTeam.end(), awayTeam.begin(), ::tolower);
            awayTeam[0] = std::toupper(awayTeam[0]);
        }
        
        return {homeTeam, awayTeam};
    }

    std::string cleanNumericValue(const std::string& value) {
        std::string cleaned = value;
        // Remove common OCR artifacts
        cleaned = std::regex_replace(cleaned, std::regex("[oO]"), "0");
        cleaned = std::regex_replace(cleaned, std::regex("[lI]"), "1");
        return cleaned;
    }

    std::vector<MatchStatistic> parseStatistics(const std::string& ocrText) {
        std::vector<MatchStatistic> statistics;
        std::vector<std::string> lines = splitLines(ocrText);
        
        // Debug: print all lines to see what OCR actually returns
        std::cerr << "\n=== ALL OCR LINES ===" << std::endl;
        for (size_t i = 0; i < lines.size(); ++i) {
            std::cerr << "Line " << i << ": '" << lines[i] << "'" << std::endl;
        }
        std::cerr << "===================" << std::endl;
        
        for (const auto& line : lines) {
            std::string lowerLine = line;
            std::transform(lowerLine.begin(), lowerLine.end(), lowerLine.begin(), ::tolower);
            
            // Try to find stat names and extract numbers from the same line
            for (const auto& statPair : statMappings) {
                if (lowerLine.find(statPair.first) != std::string::npos) {
                    MatchStatistic stat;
                    stat.statName = statPair.second;
                    
                    std::cerr << "Found stat '" << statPair.first << "' in line: '" << line << "'" << std::endl;
                    
                    // Handle percentage stats
                    if (statPair.first == "possession" || 
                        statPair.first == "passes completed" ||
                        statPair.first == "crosses completed" ||
                        statPair.first == "tackles won" ||
                        statPair.first == "headers won") {
                        
                        std::regex percentPattern("(\\d+)%");
                        std::vector<std::string> percentages;
                        
                        std::sregex_iterator iter(line.begin(), line.end(), percentPattern);
                        std::sregex_iterator end;
                        
                        for (; iter != end; ++iter) {
                            percentages.push_back(iter->str(1) + "%");
                        }
                        
                        std::cerr << "Found " << percentages.size() << " percentages" << std::endl;
                        
                        if (percentages.size() >= 2) {
                            stat.homeValue = cleanNumericValue(percentages[0]);
                            stat.awayValue = cleanNumericValue(percentages[percentages.size() - 1]);
                            statistics.push_back(stat);
                            std::cerr << "Added stat: " << stat.statName << " " << stat.homeValue << " vs " << stat.awayValue << std::endl;
                        } else if (percentages.size() == 1) {
                            // Look in adjacent lines for the second percentage
                            for (size_t j = 0; j < lines.size(); ++j) {
                                if (lines[j] != line) {
                                    std::sregex_iterator iter2(lines[j].begin(), lines[j].end(), percentPattern);
                                    if (iter2 != end) {
                                        stat.homeValue = cleanNumericValue(percentages[0]);
                                        stat.awayValue = cleanNumericValue(iter2->str(1) + "%");
                                        statistics.push_back(stat);
                                        std::cerr << "Added stat from adjacent line: " << stat.statName << " " << stat.homeValue << " vs " << stat.awayValue << std::endl;
                                        break;
                                    }
                                }
                            }
                        }
                        break;
                    } else {
                        // Extract regular numbers
                        std::regex numberPattern("(\\d+(?:\\.\\d+)?)");
                        std::vector<std::string> numbers;
                        
                        std::sregex_iterator iter(line.begin(), line.end(), numberPattern);
                        std::sregex_iterator end;
                        
                        for (; iter != end; ++iter) {
                            std::string num = iter->str(1);
                            if (num.length() <= 4) { // Skip very large numbers
                                numbers.push_back(num);
                            }
                        }
                        
                        std::cerr << "Found " << numbers.size() << " numbers: ";
                        for (const auto& num : numbers) {
                            std::cerr << num << " ";
                        }
                        std::cerr << std::endl;
                        
                        if (numbers.size() >= 2) {
                            stat.homeValue = cleanNumericValue(numbers[0]);
                            stat.awayValue = cleanNumericValue(numbers[numbers.size() - 1]);
                            statistics.push_back(stat);
                            std::cerr << "Added stat: " << stat.statName << " " << stat.homeValue << " vs " << stat.awayValue << std::endl;
                        } else if (numbers.size() == 1) {
                            // Look for the second number in adjacent lines
                            for (size_t j = 0; j < lines.size(); ++j) {
                                if (lines[j] != line) {
                                    std::sregex_iterator iter2(lines[j].begin(), lines[j].end(), numberPattern);
                                    if (iter2 != end) {
                                        std::string num2 = iter2->str(1);
                                        if (num2.length() <= 4) {
                                            stat.homeValue = cleanNumericValue(numbers[0]);
                                            stat.awayValue = cleanNumericValue(num2);
                                            statistics.push_back(stat);
                                            std::cerr << "Added stat from adjacent line: " << stat.statName << " " << stat.homeValue << " vs " << stat.awayValue << std::endl;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                        break;
                    }
                }
            }
        }
        
        return statistics;
    }

    void processScreenshot(const std::string& imagePath) {
        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) {
            std::cerr << "ERROR: Could not load image: " << imagePath << std::endl;
            return;
        }

        // Extract team names from header
        cv::Mat headerProcessed = preprocessImageForTeamNames(image);
        std::string headerOcrText = performOCR(headerProcessed);
        std::vector<std::string> headerLines = splitLines(headerOcrText);
        auto teamNames = extractTeamNames(headerLines);
        
        // Process full image for stats
        cv::Mat processed = preprocessImage(image);
        std::string ocrText = performOCR(processed);
        
        // Debug: output raw OCR text
        std::cerr << "\n=== FULL OCR TEXT ===" << std::endl;
        std::cerr << ocrText << std::endl;
        std::cerr << "====================" << std::endl;
        
        std::vector<MatchStatistic> stats = parseStatistics(ocrText);
        
        // Set team names for all stats
        for (auto& stat : stats) {
            stat.homeTeam = teamNames.first.empty() ? "Home" : teamNames.first;
            stat.awayTeam = teamNames.second.empty() ? "Away" : teamNames.second;
        }

        // Output team names
        std::cout << "HOME_TEAM:" << (teamNames.first.empty() ? "UNKNOWN" : teamNames.first) << std::endl;
        std::cout << "AWAY_TEAM:" << (teamNames.second.empty() ? "UNKNOWN" : teamNames.second) << std::endl;

        // Output statistics
        for (const auto& stat : stats) {
            std::cout << "STAT:" << stat.statName << "|" << stat.homeValue << "|" << stat.awayValue << std::endl;
        }

        std::cerr << "Total stats extracted: " << stats.size() << std::endl;

        // Save debug images
        cv::imwrite("debug_processed.png", processed);
        cv::imwrite("debug_header.png", headerProcessed);
    }
};

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <screenshot_path>" << std::endl;
        return 1;
    }

    try {
        OCRReader reader;
        reader.processScreenshot(argv[1]);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}