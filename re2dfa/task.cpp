#include "api.hpp"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <list>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#ifndef NDEBUG
#define DBG(x) x
#else
#define DBG(x)
#endif

static constexpr char FinalChar = '#';
enum class TokenKind { LParen, RParen, Star, Pipe, Symbol, Eof };
enum class ASTKind { Concatenation, Union, Iteration, Symbol };

template <typename T> std::string toString(T &&s) {
  std::string res{s.empty() ? "" : std::to_string(*s.begin())};
  for (auto it = std::next(s.begin()); it != s.end(); it++) {
    res += ";" + std::to_string(*it);
  }
  return res;
}

class Token {
public:
  Token(TokenKind kind, std::optional<char> symbol = std::nullopt)
      : m_Kind{kind}, m_SymbolValue{symbol} {}

  TokenKind getKind() const { return m_Kind; }

  bool isEOF() const {
    return m_Kind == TokenKind::Symbol && m_SymbolValue == FinalChar;
  }
  bool isPunct() const { return isOneOf(TokenKind::LParen, TokenKind::RParen); }

  [[nodiscard]] inline auto is(TokenKind kind) const -> bool {
    return m_Kind == kind;
  }

  [[nodiscard]] constexpr auto isOneOf(TokenKind tk1,
                                       TokenKind tk2) const -> bool {
    return m_Kind == tk1 || m_Kind == tk2;
  }

  template <typename... TKs>
  [[nodiscard]] constexpr auto isOneOf(TokenKind tk,
                                       TKs... other) const -> bool {
    return is(tk) || isOneOf(other...);
  }

  char getSymbol() const {
    assert(m_Kind == TokenKind::Symbol && m_SymbolValue.has_value());
    return *m_SymbolValue;
  }

private:
  TokenKind m_Kind;
  std::optional<char> m_SymbolValue;
};

std::list<Token> tokenize(std::string s) {
  s = "(" + s + ")" + FinalChar;
  std::list<Token> res;
  for (char c : s) {
    switch (c) {
    case '(':
      res.emplace_back(TokenKind::LParen);
      break;
    case ')':
      res.emplace_back(TokenKind::RParen);
      break;
    case '*':
      res.emplace_back(TokenKind::Star);
      break;
    case '|':
      res.emplace_back(TokenKind::Pipe);
      break;
    default:
      res.emplace_back(TokenKind::Symbol, c);
      break;
    }
  }

  res.emplace_back(TokenKind::Eof);
  return res;
}

std::unordered_map<int32_t, std::unordered_set<int32_t>> FollowPos;
class ASTNode {
public:
  ASTKind getKind() const { return m_Kind; }
  virtual ~ASTNode() = default;

  std::unordered_set<int32_t> FirstPos{};
  std::unordered_set<int32_t> LastPos{};
  bool Nullable{false};

protected:
  ASTNode(ASTKind kind) : m_Kind{kind} {}

private:
  ASTKind m_Kind;
};

class ConcatOpNode : public ASTNode {
public:
  ConcatOpNode(std::shared_ptr<ASTNode> lhs, std::shared_ptr<ASTNode> rhs)
      : ASTNode{ASTKind::Concatenation}, m_LHS{lhs}, m_RHS{rhs} {}

  std::shared_ptr<ASTNode> getLHS() const { return m_LHS; }
  std::shared_ptr<ASTNode> getRHS() const { return m_RHS; }

private:
  std::shared_ptr<ASTNode> m_LHS;
  std::shared_ptr<ASTNode> m_RHS;
};

class UnionOpNode : public ASTNode {
public:
  UnionOpNode(std::shared_ptr<ASTNode> lhs, std::shared_ptr<ASTNode> rhs)
      : ASTNode{ASTKind::Union}, m_LHS{std::move(lhs)}, m_RHS{std::move(rhs)} {}

  std::shared_ptr<ASTNode> getLHS() const { return m_LHS; }
  std::shared_ptr<ASTNode> getRHS() const { return m_RHS; }

private:
  std::shared_ptr<ASTNode> m_LHS;
  std::shared_ptr<ASTNode> m_RHS;
};

class IterationOpNode : public ASTNode {
public:
  IterationOpNode(std::shared_ptr<ASTNode> operand)
      : ASTNode{ASTKind::Iteration}, m_Operand{std::move(operand)} {}

  std::shared_ptr<ASTNode> getOperand() const { return m_Operand; }

private:
  std::shared_ptr<ASTNode> m_Operand;
};

class SymbolNode : public ASTNode {
public:
  bool isEmpty() const { return m_Symbol == s_EpsilonChar; }

  char getSymbol() const { return m_Symbol; }
  int32_t getID() const { return m_ID; }

  static std::shared_ptr<SymbolNode> create(char symbol) {
    static int s_IDCounter{1};
    auto [it, _] = s_SymbolsRegistry.insert(
        {s_IDCounter,
         std::shared_ptr<SymbolNode>(new SymbolNode(symbol, s_IDCounter))});
    s_SymbolNodes[symbol].insert(it->second);
    s_IDCounter++;
    return it->second;
  }

  static std::shared_ptr<SymbolNode> get(int32_t id) {
    assert(s_SymbolsRegistry.count(id) > 0);
    return s_SymbolsRegistry.at(id);
  }

  static std::unordered_set<std::shared_ptr<SymbolNode>> get(char symbol) {
    assert(s_SymbolNodes.count(symbol) > 0);
    return s_SymbolNodes.at(symbol);
  }

  static const std::shared_ptr<SymbolNode> Epsilon;

private:
  SymbolNode(char symbol, int32_t id)
      : ASTNode{ASTKind::Symbol}, m_Symbol{symbol}, m_ID{id} {}

  static constexpr char s_EpsilonChar = '$';
  static std::unordered_map<int32_t, std::shared_ptr<SymbolNode>>
      s_SymbolsRegistry;
  static std::unordered_map<char,
                            std::unordered_set<std::shared_ptr<SymbolNode>>>
      s_SymbolNodes;

  char m_Symbol;
  int32_t m_ID;
};

std::unordered_map<int32_t, std::shared_ptr<SymbolNode>>
    SymbolNode::s_SymbolsRegistry{};
std::unordered_map<char, std::unordered_set<std::shared_ptr<SymbolNode>>>
    SymbolNode::s_SymbolNodes{};
const std::shared_ptr<SymbolNode> SymbolNode::Epsilon =
    SymbolNode::create(SymbolNode::s_EpsilonChar);

static std::unordered_map<TokenKind, int32_t> s_OpPrecedence{
    {TokenKind::Star, 30},
    {TokenKind::Pipe, 10},
};

class Parser {
public:
  Parser(std::list<Token> tokens)
      : m_Tokens{std::move(tokens)}, m_CurTokIterator{m_Tokens.begin()} {}

  auto parse() -> std::shared_ptr<ASTNode> { return parseExpr(); }

private:
  auto parseCompoundExpression() -> std::shared_ptr<ASTNode> {
    auto tok{curTok()};
    switch (tok.getKind()) {
    case TokenKind::Symbol:
      consumeToken();
      return SymbolNode::create(tok.getSymbol());
    case TokenKind::LParen:
      return parseParenExpr();
    default:
      return SymbolNode::Epsilon;
    }
  }

  bool isBinOpTok() { return curTok().is(TokenKind::Pipe); }

  auto parseIteration() -> std::shared_ptr<ASTNode> {
    auto lhs{parseCompoundExpression()};
    while (curTok().is(TokenKind::Star)) {
      consumeToken();
      lhs = std::make_shared<IterationOpNode>(std::move(lhs));
    }
    return lhs;
  }

  auto parseConcat() -> std::shared_ptr<ASTNode> {
    auto lhs{parseIteration()};
    while (
        !curTok().isOneOf(TokenKind::Pipe, TokenKind::RParen, TokenKind::Eof)) {
      auto rhs{parseIteration()};
      lhs = std::make_shared<ConcatOpNode>(std::move(lhs), std::move(rhs));
    }
    return lhs;
  }

  auto parseExpr() -> std::shared_ptr<ASTNode> {
    auto lhs{parseConcat()};
    while (curTok().is(TokenKind::Pipe)) {
      consumeToken();
      auto rhs{parseConcat()};
      lhs = std::make_shared<UnionOpNode>(std::move(lhs), std::move(rhs));
    }
    return lhs;
  }

  // auto parseBiinOpRHS(std::shared_ptr<ASTNode> lhs,
  //                     int32_t prevPrecedence = 0) -> std::shared_ptr<ASTNode>
  //                     {
  //   while (true) {
  //     if (curTok().is(TokenKind::Star)) {
  //       consumeToken();
  //       lhs = std::make_shared<IterationOpNode>(std::move(lhs));
  //     }
  //
  //     // precedence or -1 if not a binop
  //     auto tokPrecedence{getTokPrecedence(curTok())};
  //     if (tokPrecedence < prevPrecedence) {
  //       return lhs;
  //     }
  //
  //     if (!isBinOpTok()) {
  //       auto rhs{parseCompoundExpression()};
  //       // we need every time before combining anything
  //       if (curTok().is(TokenKind::Star)) {
  //         consumeToken();
  //         rhs = std::make_shared<IterationOpNode>(std::move(rhs));
  //       }
  //       lhs = std::make_shared<ConcatOpNode>(std::move(lhs), std::move(rhs));
  //       continue;
  //     }
  //
  //     auto binOp{curTok()};
  //     consumeToken();
  //     auto rhs{parseCompoundExpression()};
  //
  //     // now: lhs binOp rhs unparsed
  //
  //     if (curTok().is(TokenKind::Star)) {
  //       consumeToken();
  //       rhs = std::make_shared<IterationOpNode>(std::move(rhs));
  //     }
  //
  //     auto nextPrecedence{getTokPrecedence(curTok())};
  //     // if associates to the right: lhs binOp (rhs lookahead unparsed)
  //     if (tokPrecedence < nextPrecedence) {
  //       rhs = parseBinOpRHS(std::move(rhs), tokPrecedence + 1);
  //     }
  //
  //     if (binOp.is(TokenKind::Pipe)) {
  //       lhs = std::make_shared<UnionOpNode>(std::move(lhs), std::move(rhs));
  //     } else {
  //       lhs = std::make_shared<ConcatOpNode>(std::move(lhs), std::move(rhs));
  //     }
  //   }
  // }

  auto parseParenExpr() -> std::shared_ptr<ASTNode> {
    consumeToken();

    auto result{parseExpr()};
    if (!result) {
    }

    consumeToken();
    return result;
  }

  void consumeToken() { m_CurTokIterator++; }

  Token &curTok() { return *m_CurTokIterator; }

  int32_t getTokPrecedence(const Token &tok) const {
    auto it = s_OpPrecedence.find(tok.getKind());
    if (tok.isEOF() || tok.isPunct()) {
      return -1;
    }
    if (it == s_OpPrecedence.end()) {
      return 20;
    }
    return it->second;
  }

private:
  std::list<Token> m_Tokens;
  std::list<Token>::iterator m_CurTokIterator;
};

std::shared_ptr<ASTNode> parse(const std::list<Token> &tokens) {
  Parser parser(tokens);
  return parser.parse();
}

class ASTVisitor {
public:
  void visit(const std::shared_ptr<ASTNode> &node) {
    switch (node->getKind()) {
    case ASTKind::Iteration:
      visitImpl(std::dynamic_pointer_cast<IterationOpNode>(node));
      break;
    case ASTKind::Concatenation:
      visitImpl(std::dynamic_pointer_cast<ConcatOpNode>(node));
      break;
    case ASTKind::Union:
      visitImpl(std::dynamic_pointer_cast<UnionOpNode>(node));
      break;
    case ASTKind::Symbol:
      visitImpl(std::dynamic_pointer_cast<SymbolNode>(node));
      break;
    }
  }

protected:
  std::string info(const std::shared_ptr<ASTNode> &node) {
    std::stringstream ss;
    ss << " fp=" << toString(node->FirstPos)
       << ",lp=" << toString(node->LastPos);
    return ss.str();
  }

  virtual void visitImpl(std::shared_ptr<IterationOpNode> node) {
    std::cout << std::string(depth, ' ') << "iteration" << info(node)
              << std::endl;
    depth++;
    visit(node->getOperand());
    depth--;
  }
  virtual void visitImpl(std::shared_ptr<ConcatOpNode> node) {
    std::cout << std::string(depth, ' ') << "concatenation" << info(node)
              << std::endl;
    depth++;
    visit(node->getLHS());
    visit(node->getRHS());
    depth--;
  }
  virtual void visitImpl(std::shared_ptr<UnionOpNode> node) {
    std::cout << std::string(depth, ' ') << "union" << info(node) << std::endl;
    depth++;
    visit(node->getLHS());
    visit(node->getRHS());
    depth--;
  }
  virtual void visitImpl(std::shared_ptr<SymbolNode> node) {
    std::cout << std::string(depth, ' ') << "symbol: " << node->getSymbol()
              << " info:" << info(node) << std::endl;
  }

private:
  int depth = 0;
};

class PosResolver : public ASTVisitor {
  virtual void visitImpl(std::shared_ptr<IterationOpNode> node) {
    ASTVisitor::visit(node->getOperand());
    node->Nullable = true;
    node->FirstPos = node->getOperand()->FirstPos;
    node->LastPos = node->getOperand()->LastPos;
    for (auto lp : node->LastPos) {
      FollowPos[lp].insert(node->FirstPos.begin(), node->FirstPos.end());
    }
  }
  virtual void visitImpl(std::shared_ptr<ConcatOpNode> node) {
    ASTVisitor::visit(node->getLHS());
    ASTVisitor::visit(node->getRHS());
    node->Nullable = node->getLHS()->Nullable && node->getRHS()->Nullable;
    node->FirstPos.insert(node->getLHS()->FirstPos.begin(),
                          node->getLHS()->FirstPos.end());
    node->LastPos.insert(node->getRHS()->LastPos.begin(),
                         node->getRHS()->LastPos.end());
    if (node->getLHS()->Nullable) {
      node->FirstPos.insert(node->getRHS()->FirstPos.begin(),
                            node->getRHS()->FirstPos.end());
    }
    if (node->getRHS()->Nullable) {
      node->LastPos.insert(node->getLHS()->LastPos.begin(),
                           node->getLHS()->LastPos.end());
    }

    for (auto lp : node->getLHS()->LastPos) {
      FollowPos[lp].insert(node->getRHS()->FirstPos.begin(),
                           node->getRHS()->FirstPos.end());
    }
  }
  virtual void visitImpl(std::shared_ptr<UnionOpNode> node) {
    ASTVisitor::visit(node->getLHS());
    ASTVisitor::visit(node->getRHS());
    node->Nullable = node->getLHS()->Nullable || node->getRHS()->Nullable;
    node->FirstPos.insert(node->getLHS()->FirstPos.begin(),
                          node->getLHS()->FirstPos.end());
    node->FirstPos.insert(node->getRHS()->FirstPos.begin(),
                          node->getRHS()->FirstPos.end());
    node->LastPos.insert(node->getLHS()->LastPos.begin(),
                         node->getLHS()->LastPos.end());
    node->LastPos.insert(node->getRHS()->LastPos.begin(),
                         node->getRHS()->LastPos.end());
  }
  virtual void visitImpl(std::shared_ptr<SymbolNode> node) {
    node->FirstPos.insert(node->getID());
    node->LastPos.insert(node->getID());
    node->Nullable = node->isEmpty();
  }
};

using State = std::set<int32_t>;

bool isStateFinal(const State &state) {
  assert(SymbolNode::get(FinalChar).size() == 1);
  return std::find_if(state.begin(), state.end(), [](const auto &pos) {
           return (*SymbolNode::get(FinalChar).begin())->getID() == pos;
         }) != state.end();
}

DFA re2dfa(const std::string &s) {
  if (s.empty()) {
    DFA res(Alphabet("e"));
    res.create_state("q0", true);
    return res;
  }

  DFA res = DFA(Alphabet(s));

  auto tokens = tokenize(s);
  auto ast = parse(tokens);
  PosResolver resolver;
  resolver.visit(ast);

  DBG({
    ASTVisitor visitor;
    visitor.visit(ast);

    std::cout << "flp:" << std::endl;
    for (auto [k, v] : FollowPos) {
      std::cout << k << "(" << SymbolNode::get(k)->getSymbol()
                << "): " << toString(v) << std::endl;
    }
  });

  Alphabet alphabet(s);
  State q0{ast->FirstPos.begin(), ast->FirstPos.end()};
  std::list<State> Q{q0};
  std::set<State> marked;
  while (!Q.empty()) {
    auto R = Q.front();
    if (marked.count(R)) {
      Q.pop_front();
      continue;
    }

    res.create_state(toString(R), isStateFinal(R));
    marked.insert(R);

    for (auto sym : alphabet) {
      State S;
      for (auto p : SymbolNode::get(sym)) {
        if (R.count(p->getID())) {
          auto &&fp = FollowPos[p->getID()];
          S.insert(fp.begin(), fp.end());
        }
      }
      if (!S.empty()) {
        if (std::find(Q.begin(), Q.end(), S) == Q.end())
          Q.push_back(S);
        res.create_state(toString(S), isStateFinal(S));
        res.set_trans(toString(R), sym, toString(S));
      }
    }
    Q.pop_front();
  }

  res.set_initial(toString(q0));
  return res;
}
